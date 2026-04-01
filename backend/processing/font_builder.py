"""
font_builder.py — Assemble OTF font from vectorized glyphs using fonttools.

Coordinate system:
  - Source: SVG pixel space, y increases downward, origin top-left of upscaled glyph image
  - Target: Font UPM space (1000 units), y increases upward, baseline at y=0

Transform for each point:
  font_x = svg_x * scale_x + x_offset
  font_y = -(svg_y * scale_y) + y_offset   # negation = Y-axis flip
"""

import io
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fontTools.fontBuilder import FontBuilder
from fontTools.feaLib.builder import addOpenTypeFeatures
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont

from template import TEMPLATE_SPEC, MM_TO_PX

# ---------------------------------------------------------------------------
# Font metric constants (in UPM = 1000)
# ---------------------------------------------------------------------------

UPM = 1000
ASCENDER = 800
CAP_HEIGHT = 700
WIN_ASCENT = 900
WIN_DESCENT = 200
DEFAULT_ADVANCE_WIDTH = 600

# Derive vertical metrics from TEMPLATE_SPEC so glyph scale is consistent
_cap_to_base  = TEMPLATE_SPEC["guideline_baseline_ratio"] - TEMPLATE_SPEC["guideline_top_ratio"]
_x_to_base    = TEMPLATE_SPEC["guideline_baseline_ratio"] - TEMPLATE_SPEC["guideline_xheight_ratio"]
_base_to_desc = TEMPLATE_SPEC["guideline_bottom_ratio"]   - TEMPLATE_SPEC["guideline_baseline_ratio"]

X_HEIGHT  = int(_x_to_base  / _cap_to_base * CAP_HEIGHT)    # ≈ 368
DESCENDER = -int(_base_to_desc / _cap_to_base * CAP_HEIGHT)  # ≈ -221

# Cell-level scale: UPM per original pixel (same for ALL glyphs)
CELL_BASELINE_Y_PX = int(TEMPLATE_SPEC["cell_height_mm"] * MM_TO_PX * _cap_to_base)
CELL_SCALE = CAP_HEIGHT / CELL_BASELINE_Y_PX


# ---------------------------------------------------------------------------
# Glyph name helpers
# ---------------------------------------------------------------------------

CHAR_TO_GLYPH_NAME: Dict[str, str] = {
    ' ': 'space',
    '!': 'exclam', '"': 'quotedbl', '#': 'numbersign', '&': 'ampersand',
    "'": 'quotesingle', '(': 'parenleft', ')': 'parenright', ',': 'comma',
    '-': 'hyphen', '.': 'period', '/': 'slash', ':': 'colon',
    ';': 'semicolon', '?': 'question', '@': 'at', '%': 'percent',
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

def char_to_glyph_name(char: str, slot: int = 0) -> str:
    """Map character + slot to an OpenType glyph name."""
    base = CHAR_TO_GLYPH_NAME.get(char, char)
    if slot == 0:
        return base
    return f"{base}.alt{slot}"


def char_to_unicode(char: str) -> Optional[int]:
    """Return Unicode codepoint for a character, or None for non-mappable."""
    if len(char) == 1:
        return ord(char)
    return None


# ---------------------------------------------------------------------------
# SVG path → fonttools Pen
# ---------------------------------------------------------------------------

def _parse_svg_path_commands(d: str) -> List[Tuple]:
    """
    Parse an SVG path 'd' attribute into a list of (command, [args]) tuples.
    Handles absolute M, L, C, Z commands (vtracer output).
    """
    # Tokenize path data
    tokens = re.findall(r'[MLCZmlcz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', d)
    commands = []
    i = 0
    current_cmd = None
    args = []

    def flush():
        if current_cmd:
            commands.append((current_cmd, args[:]))

    while i < len(tokens):
        tok = tokens[i]
        if tok in 'MLCZmlcz':
            flush()
            args = []
            current_cmd = tok
            if tok in 'Zz':
                commands.append((tok, []))
                current_cmd = None
        else:
            args.append(float(tok))
        i += 1

    flush()
    return commands


def _draw_svg_paths_to_pen(
    pen,
    svg_paths: List[str],
    svg_width: int,
    svg_height: int,
    baseline_y_in_svg: int,     # in ORIGINAL (non-upscaled) pixel space
    target_height: int = CAP_HEIGHT,   # unused, kept for compat
    is_lowercase: bool = False,        # unused, kept for compat
    upscale_factor: float = 1.0,
) -> int:
    """
    Draw SVG paths into a fonttools pen, applying coordinate transform.

    Uses a cell-level scale (CELL_SCALE) derived from TEMPLATE_SPEC so that
    ALL glyphs share the same coordinate system — descenders, punctuation, and
    letters all align correctly relative to the baseline.

    Returns the advance width (in UPM units).
    """
    if not svg_paths or svg_width == 0 or svg_height == 0:
        return DEFAULT_ADVANCE_WIDTH

    # coord_scale: converts upscaled SVG pixels → UPM units
    # baseline_y_in_svg is in original pixel space; CELL_SCALE converts it to UPM
    coord_scale = CELL_SCALE / upscale_factor
    y_offset = baseline_y_in_svg * CELL_SCALE   # UPM position of the baseline

    advance_width = int(svg_width * coord_scale) + 60  # proportional advance + side bearings

    pen.beginPath = getattr(pen, 'beginPath', None)

    for path_d in svg_paths:
        commands = _parse_svg_path_commands(path_d)
        if not commands:
            continue

        started = False
        current_x, current_y = 0.0, 0.0

        for cmd, args in commands:
            if cmd == 'M':
                if started:
                    pen.endPath()
                sx, sy = args[0], args[1]
                fx = sx * coord_scale
                fy = -(sy * coord_scale) + y_offset
                pen.moveTo((fx, fy))
                current_x, current_y = sx, sy
                started = True

            elif cmd == 'L':
                sx, sy = args[0], args[1]
                fx = sx * coord_scale
                fy = -(sy * coord_scale) + y_offset
                pen.lineTo((fx, fy))
                current_x, current_y = sx, sy

            elif cmd == 'C':
                # Cubic bezier: C x1 y1 x2 y2 x y
                for j in range(0, len(args), 6):
                    if j + 5 >= len(args):
                        break
                    x1, y1 = args[j], args[j + 1]
                    x2, y2 = args[j + 2], args[j + 3]
                    x, y = args[j + 4], args[j + 5]
                    pen.curveTo(
                        (x1 * coord_scale, -(y1 * coord_scale) + y_offset),
                        (x2 * coord_scale, -(y2 * coord_scale) + y_offset),
                        (x * coord_scale, -(y * coord_scale) + y_offset),
                    )
                    current_x, current_y = x, y

            elif cmd in ('Z', 'z'):
                if started:
                    pen.closePath()
                    started = False

        if started:
            pen.endPath()

    return advance_width


# ---------------------------------------------------------------------------
# Feature code generation
# ---------------------------------------------------------------------------

def _build_fea_code(alternates: Dict[str, List[str]], all_glyph_names: List[str] = None) -> str:
    """
    Build OpenType feature code (.fea) for calt and ss01/ss02.

    Cycling works for every occurrence of a letter, not just adjacent pairs.
    Rules look back up to MAX_DIST positions so that "banana" produces
    b a(form0) n a(form1) n a(form0) — alternating on every occurrence.

    Args:
        alternates: {base_glyph_name: [alt1_name, alt2_name, ...]}
        all_glyph_names: full glyph order (used to build @other classes)
    """
    MAX_DIST = 8  # look back up to 8 glyphs for the previous occurrence

    if all_glyph_names is None:
        all_glyph_names = ["space"]
        for base, alts in alternates.items():
            all_glyph_names.append(base)
            all_glyph_names.extend(alts)

    lines = []

    # Per-letter @other glyph classes (everything except this letter's forms)
    for base, alts in alternates.items():
        if not alts:
            continue
        forms_set = set([base] + alts)
        other = [g for g in all_glyph_names
                 if g not in forms_set and g != ".notdef" and not g[0].isdigit()]
        if other:
            lines.append(f"@{base}_other = [{' '.join(other)}];")
    lines.append("")

    # calt feature: distance-based cycling rules ordered shortest→longest
    # Rule: sub {prev_form} {@other}*(dist-1) {base}' by {next_form};
    # OpenType processes left-to-right; shortest match fires first (most recent occurrence wins).
    lines.append("feature calt {")
    for base, alts in alternates.items():
        if not alts:
            continue
        all_forms = [base] + alts
        n = len(all_forms)
        forms_set = set(all_forms)
        other = [g for g in all_glyph_names
                 if g not in forms_set and g != ".notdef" and not g[0].isdigit()]
        has_other = bool(other)

        for dist in range(1, MAX_DIST + 1):
            if dist > 1 and not has_other:
                break
            middle = f"@{base}_other " * (dist - 1)
            for i, prev_form in enumerate(all_forms):
                next_form = all_forms[(i + 1) % n]
                lines.append(f"    sub {prev_form} {middle}{base}' by {next_form};")
    lines.append("} calt;")
    lines.append("")

    # ss01: force all to first alternate
    if any(alts for alts in alternates.values()):
        lines.append("feature ss01 {")
        for base, alts in alternates.items():
            if alts:
                lines.append(f"    sub {base} by {alts[0]};")
        lines.append("} ss01;")
        lines.append("")

    # ss02: force all to second alternate (if available)
    if any(len(alts) >= 2 for alts in alternates.values()):
        lines.append("feature ss02 {")
        for base, alts in alternates.items():
            if len(alts) >= 2:
                lines.append(f"    sub {base} by {alts[1]};")
        lines.append("} ss02;")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main font builder
# ---------------------------------------------------------------------------

@dataclass
class GlyphData:
    char: str
    slot: int
    glyph_name: str
    svg_paths: List[str]
    svg_width: int
    svg_height: int
    baseline_y_in_svg: int   # in original (non-upscaled) pixel space
    is_lowercase: bool
    upscale_factor: float = 1.0


def build_otf(
    glyphs: List[GlyphData],
    font_name: str,
    font_style: str = "Regular",
    letter_spacing: int = 0,
    space_width: int = DEFAULT_ADVANCE_WIDTH,
) -> bytes:
    """
    Assemble an OTF font from a list of vectorized glyphs.

    Args:
        glyphs: list of GlyphData, one per accepted glyph
        font_name: the font family name (user-supplied)
        font_style: style name (default "Regular")
        letter_spacing: extra UPM units added to every glyph's advance width (tracking)
        space_width: advance width for the space glyph in UPM units

    Returns:
        Raw OTF bytes.
    """
    fb = FontBuilder(UPM, isTTF=False)

    # Build glyph order
    glyph_order = [".notdef", "space"]
    seen = set(glyph_order)
    for g in glyphs:
        if g.glyph_name not in seen:
            glyph_order.append(g.glyph_name)
            seen.add(g.glyph_name)

    fb.setupGlyphOrder(glyph_order)

    # Character map (only primary glyphs / slot==0 map to Unicode)
    cmap: Dict[int, str] = {0x0020: "space"}
    for g in glyphs:
        if g.slot == 0:
            cp = char_to_unicode(g.char)
            if cp is not None:
                cmap[cp] = g.glyph_name

    fb.setupCharacterMap(cmap)

    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.misc.psCharStrings import T2CharString

    private_dict = {"defaultWidthX": DEFAULT_ADVANCE_WIDTH, "nominalWidthX": 0}
    charstrings: Dict[str, any] = {}
    metrics: Dict[str, Tuple[int, int]] = {
        ".notdef": (500, 0),
        "space": (space_width, 0),
    }

    # .notdef
    notdef_pen = T2CharStringPen(500, glyphSet=None)
    charstrings[".notdef"] = notdef_pen.getCharString()

    # space — encode the width explicitly as the first element of the CFF program.
    # CFF rule: if the charstring begins with a number before any drawing op, that number
    # is (advance - nominalWidthX). With nominalWidthX=0, encoding space_width directly
    # gives the correct advance. Using just ['endchar'] would fall back to defaultWidthX=600.
    from fontTools.misc.psCharStrings import T2CharString as _T2CS
    _space_cs = _T2CS()
    _space_cs.program = [space_width, "endchar"]
    charstrings["space"] = _space_cs
    print(f"[font_builder] space_width={space_width}  letter_spacing={letter_spacing}  program={_space_cs.program}")

    for g in glyphs:
        cs, advance = _build_charstring_from_svg(
            g.svg_paths, g.svg_width, g.svg_height,
            g.baseline_y_in_svg, g.is_lowercase,
            upscale_factor=g.upscale_factor,
        )
        # Keep CFF charstring width consistent with hmtx (both include letter_spacing)
        if letter_spacing != 0 and cs.program and isinstance(cs.program[0], (int, float)):
            cs.program[0] = advance + letter_spacing
        charstrings[g.glyph_name] = cs
        metrics[g.glyph_name] = (advance + letter_spacing, 0)

    fb.setupCFF(
        psName=font_name,
        fontInfo={"version": "1.0", "FullName": f"{font_name} {font_style}",
                  "FamilyName": font_name, "Weight": font_style},
        charStringsDict=charstrings,
        privateDict=private_dict,
    )

    fb.setupHorizontalMetrics(metrics)

    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)

    fb.setupNameTable({
        "familyName": font_name,
        "styleName": font_style,
    })

    fb.setupOS2(
        sTypoAscender=ASCENDER,
        sTypoDescender=DESCENDER,
        sTypoLineGap=0,
        usWinAscent=WIN_ASCENT,
        usWinDescent=WIN_DESCENT,
        sxHeight=X_HEIGHT,
        sCapHeight=CAP_HEIGHT,
        fsType=0,
    )

    fb.setupPost()
    fb.setupHead(unitsPerEm=UPM)

    # Add OpenType features (calt, ss01, ss02)
    fea_warning = None
    alternates = _collect_alternates(glyphs)
    if alternates:
        fea_code = _build_fea_code(alternates, all_glyph_names=glyph_order)
        try:
            addOpenTypeFeatures(fb.font, io.StringIO(fea_code))
        except Exception as e:
            fea_warning = str(e)
            print(f"Warning: feaLib error (features skipped): {e}")

    buf = io.BytesIO()
    fb.font.save(buf)

    # Verify actual advance widths in the saved font
    buf.seek(0)
    _verify = TTFont(buf)
    _hmtx = _verify["hmtx"].metrics
    print(f"[font_builder] saved hmtx space={_hmtx.get('space')}  .notdef={_hmtx.get('.notdef')}")
    buf.seek(0)

    return buf.getvalue(), fea_warning


def otf_to_woff2(otf_bytes: bytes) -> bytes:
    """Convert raw OTF bytes to WOFF2."""
    font = TTFont(io.BytesIO(otf_bytes))
    buf = io.BytesIO()
    font.flavor = "woff2"
    font.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _DeferredPen:
    """Collects pen operations for later replay (not actually used — see charstring builder)."""
    def __init__(self, name, store):
        self.name = name
        self.store = store
        self.ops = []
        self.store[name] = self

    def moveTo(self, pt): self.ops.append(("moveTo", pt))
    def lineTo(self, pt): self.ops.append(("lineTo", pt))
    def curveTo(self, *pts): self.ops.append(("curveTo", pts))
    def closePath(self): self.ops.append(("closePath",))
    def endPath(self): self.ops.append(("endPath",))


def _build_charstring_from_svg(
    svg_paths: List[str],
    svg_width: int,
    svg_height: int,
    baseline_y_in_svg: int,
    is_lowercase: bool,
    upscale_factor: float = 1.0,
) -> Tuple["T2CharString", int]:
    """Build a CFF T2CharString by drawing SVG paths. Returns (charstring, advance_width)."""
    from fontTools.pens.t2CharStringPen import T2CharStringPen

    coord_scale = CELL_SCALE / upscale_factor if upscale_factor > 0 else CELL_SCALE
    advance = int(svg_width * coord_scale) + 60 if svg_width > 0 else DEFAULT_ADVANCE_WIDTH

    pen = T2CharStringPen(advance, glyphSet=None)

    _draw_svg_paths_to_pen(
        pen, svg_paths, svg_width, svg_height,
        baseline_y_in_svg, upscale_factor=upscale_factor,
    )

    return pen.getCharString(), advance


def _build_notdef_charstring() -> "T2CharString":
    """Build a simple box glyph for .notdef."""
    from fontTools.misc.psCharStrings import T2CharString

    cs = T2CharString()
    w = 500
    # Draw a simple rectangle outline
    cs.program = [
        w,          # width
        50, 0,      # rmoveto: move to (50, 0)
        0, 700,     # rlineto: up
        400, 0,     # rlineto: right
        0, -700,    # rlineto: down
        -400, 0,    # rlineto: left (close)
        # Inner box
        50, 50, "rmoveto",
        300, 0, "rlineto",
        0, 600, "rlineto",
        -300, 0, "rlineto",
        0, -600, "rlineto",
        "endchar",
    ]
    # Simpler version:
    cs.program = [
        w,
        "endchar",
    ]
    return cs


def _collect_alternates(glyphs: List[GlyphData]) -> Dict[str, List[str]]:
    """
    Build the alternates map: {base_glyph_name: [alt1_name, alt2_name, ...]}.
    Only includes characters that actually have alternates in the approved glyph set.
    """
    primaries = {g.char: g.glyph_name for g in glyphs if g.slot == 0}
    alts: Dict[str, List[str]] = {}

    for g in glyphs:
        if g.slot > 0 and g.char in primaries:
            base = primaries[g.char]
            if base not in alts:
                alts[base] = []
            alts[base].append(g.glyph_name)

    # Sort alts by slot number
    for base in alts:
        alts[base].sort(key=lambda n: int(n.split("alt")[-1]) if "alt" in n else 0)

    return alts
