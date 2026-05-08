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

# Canvas geometry (formerly TEMPLATE_SPEC). The drawing canvas uses these
# same guideline ratios so glyphs come in with consistent baselines.
CELL_HEIGHT_MM = 24.0
MM_TO_PX = 11.811                       # 300 dpi
GUIDELINE_TOP_RATIO       = 0.15        # CAP line
GUIDELINE_XHEIGHT_RATIO   = 0.42
GUIDELINE_BASELINE_RATIO  = 0.72
GUIDELINE_BOTTOM_RATIO    = 0.90        # descender

# ---------------------------------------------------------------------------
# Font metric constants (in UPM = 1000)
# ---------------------------------------------------------------------------

UPM = 1000
ASCENDER = 800
CAP_HEIGHT = 700
WIN_ASCENT = 900
WIN_DESCENT = 200
DEFAULT_ADVANCE_WIDTH = 600

# Derive vertical metrics from the guideline ratios so the canvas and the
# font share one coordinate system.
_cap_to_base  = GUIDELINE_BASELINE_RATIO - GUIDELINE_TOP_RATIO
_x_to_base    = GUIDELINE_BASELINE_RATIO - GUIDELINE_XHEIGHT_RATIO
_base_to_desc = GUIDELINE_BOTTOM_RATIO   - GUIDELINE_BASELINE_RATIO

X_HEIGHT  = int(_x_to_base  / _cap_to_base * CAP_HEIGHT)
DESCENDER = -int(_base_to_desc / _cap_to_base * CAP_HEIGHT)

# Cell-level scale: UPM per original canvas pixel (same for every glyph).
CELL_BASELINE_Y_PX = int(CELL_HEIGHT_MM * MM_TO_PX * _cap_to_base)
CELL_SCALE = CAP_HEIGHT / CELL_BASELINE_Y_PX


# ---------------------------------------------------------------------------
# Glyph name helpers
# ---------------------------------------------------------------------------

CHAR_TO_GLYPH_NAME: Dict[str, str] = {
    ' ': 'space',
    '!': 'exclam', '"': 'quotedbl', '#': 'numbersign', '$': 'dollar',
    '%': 'percent', '&': 'ampersand', "'": 'quotesingle',
    '(': 'parenleft', ')': 'parenright', '*': 'asterisk', '+': 'plus',
    ',': 'comma', '-': 'hyphen', '.': 'period', '/': 'slash',
    ':': 'colon', ';': 'semicolon', '<': 'less', '=': 'equal',
    '>': 'greater', '?': 'question', '@': 'at',
    '[': 'bracketleft', '\\': 'backslash', ']': 'bracketright',
    '^': 'asciicircum', '_': 'underscore', '`': 'grave',
    '{': 'braceleft', '|': 'bar', '}': 'braceright', '~': 'asciitilde',
    '–': 'endash', '—': 'emdash',
    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine',
}

def char_to_glyph_name(char: str, slot: int = 0, form: str = "iso") -> str:
    """
    Map character + slot + cursive form to an OpenType glyph name.

    Print mode glyphs use slot suffixes ('a', 'a.alt1', 'a.alt2').
    Cursive positional variants use form suffixes ('a.init', 'a.medi',
    'a.fina'); the iso form keeps the bare base name so cmap can map it
    directly to the Unicode codepoint.
    """
    base = CHAR_TO_GLYPH_NAME.get(char, char)
    if form != "iso":
        return f"{base}.{form}"
    if slot != 0:
        return f"{base}.alt{slot}"
    return base


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


def _bearing_offsets(
    form: str,
    coord_scale: float,
    svg_width: int = 0,
    entry_x: Optional[float] = None,
    exit_x:  Optional[float] = None,
) -> Tuple[float, int]:
    """
    Return (x_offset, advance_extra) for a glyph's positional form.

    Two paths:

    * If `entry_x` and/or `exit_x` are given (post-xShift canvas px),
      compute bearings from the user-set connection points. This lets each
      letter dial in its own connection x to match its natural width.
        - x_offset places `entry_x` at font x=0 (medi/fina)
        - advance places `exit_x` at font x=advance (init/medi)

    * Otherwise fall back to the canvas-PAD assumption (legacy behaviour).
    """
    pad_upm = CANVAS_PAD * coord_scale

    def x_for_entry(ex: float) -> float:
        return -float(ex) * coord_scale

    def adv_for_exit(ex: float, x_off: float) -> int:
        # advance = font x of exit point = exit_x_canvas * cs + x_offset
        return int(float(ex) * coord_scale + x_off)

    if form == "init":
        x_off = 0.0
        if exit_x is not None and svg_width > 0:
            return x_off, adv_for_exit(exit_x, x_off) - int(svg_width * coord_scale)
        return x_off, -int(pad_upm)

    if form == "medi":
        x_off = x_for_entry(entry_x) if entry_x is not None else -pad_upm
        if exit_x is not None and svg_width > 0:
            return x_off, adv_for_exit(exit_x, x_off) - int(svg_width * coord_scale)
        return x_off, -int(2 * pad_upm)

    if form == "fina":
        x_off = x_for_entry(entry_x) if entry_x is not None else -pad_upm
        return x_off, 0

    # iso (and print mode default): canvas PAD + historical +60 UPM trailing
    # breathing room between standalone letters.
    return 0.0, 60


def _draw_svg_paths_to_pen(
    pen,
    svg_paths: List[str],
    svg_width: int,
    svg_height: int,
    baseline_y_in_svg: int,     # in ORIGINAL (non-upscaled) pixel space
    target_height: int = CAP_HEIGHT,   # unused, kept for compat
    is_lowercase: bool = False,        # unused, kept for compat
    upscale_factor: float = 1.0,
    form: str = "iso",
    entry_x: Optional[float] = None,
    exit_x:  Optional[float] = None,
) -> int:
    """
    Draw SVG paths into a fonttools pen, applying coordinate transform and
    cursive bearing adjustments per `form`.

    Returns the advance width (in UPM units).
    """
    if not svg_paths or svg_width == 0 or svg_height == 0:
        return DEFAULT_ADVANCE_WIDTH

    coord_scale = CELL_SCALE / upscale_factor
    y_offset = baseline_y_in_svg * CELL_SCALE

    x_offset, adv_extra = _bearing_offsets(form, coord_scale, svg_width, entry_x, exit_x)
    advance_width = int(svg_width * coord_scale) + adv_extra

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
                fx = sx * coord_scale + x_offset
                fy = -(sy * coord_scale) + y_offset
                pen.moveTo((fx, fy))
                current_x, current_y = sx, sy
                started = True

            elif cmd == 'L':
                sx, sy = args[0], args[1]
                fx = sx * coord_scale + x_offset
                fy = -(sy * coord_scale) + y_offset
                pen.lineTo((fx, fy))
                current_x, current_y = sx, sy

            elif cmd == 'C':
                for j in range(0, len(args), 6):
                    if j + 5 >= len(args):
                        break
                    x1, y1 = args[j], args[j + 1]
                    x2, y2 = args[j + 2], args[j + 3]
                    x, y = args[j + 4], args[j + 5]
                    pen.curveTo(
                        (x1 * coord_scale + x_offset, -(y1 * coord_scale) + y_offset),
                        (x2 * coord_scale + x_offset, -(y2 * coord_scale) + y_offset),
                        (x * coord_scale + x_offset, -(y * coord_scale) + y_offset),
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
    MAX_DIST = 12  # look back up to 12 glyphs for the previous occurrence
                    # — covers most words plus typical inter-word spacing,
                    # so even repeated letters in a long phrase keep cycling
                    # through their variants.

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


def _build_cursive_fea_code(
    positional: Dict[str, Dict[str, str]],
    available_glyphs: Optional[List[str]] = None,
) -> str:
    """
    Build OpenType `calt` rules that swap a lowercase letter for its
    initial / medial / final form based on whether it's surrounded by
    other joining letters.

    `positional` maps each base iso glyph name to the names of its
    positional variants, e.g.:

        {"a": {"init": "a.init", "medi": "a.medi", "fina": "a.fina"}, ...}

    A letter without any of the three positional forms is skipped (stays
    as the iso form regardless of position). Latin doesn't have native
    init/medi/fina shaping, so we use `calt` which is on by default.
    """
    if not positional:
        return ""

    chars = sorted(positional.keys())
    init_lhs, init_rhs = [], []
    medi_lhs, medi_rhs = [], []
    fina_lhs, fina_rhs = [], []
    for c in chars:
        forms = positional[c]
        if "init" in forms:
            init_lhs.append(c); init_rhs.append(forms["init"])
        if "medi" in forms:
            medi_lhs.append(c); medi_rhs.append(forms["medi"])
        if "fina" in forms:
            fina_lhs.append(c); fina_rhs.append(forms["fina"])

    if not (init_lhs or medi_lhs or fina_lhs):
        return ""

    # @joining must include every lowercase letter present in the font, not
    # just letters with positional forms drawn — otherwise pairs like 'ab'
    # where only 'a' has positional forms wouldn't join (the engine would
    # see 'b' as a word boundary). Capitals, digits, punctuation and space
    # stay out so they correctly act as word boundaries.
    #
    # CRITICAL: @joining must also include the positional-form glyph names
    # (a.init, a.medi, …). The calt lookups run in order — once `b → b.medi`
    # has fired, the next lookup checking "is position N preceded by a
    # joining letter?" needs to count `b.medi` as joining, not just `b`.
    # Without this, only the medi rule fires; init and fina would silently
    # fail to chain past it.
    available = set(available_glyphs or [])
    if available:
        joining_iso = sorted(c for c in 'abcdefghijklmnopqrstuvwxyz' if c in available)
    else:
        joining_iso = sorted(chars)
    if not joining_iso:
        return ""

    joining_all = list(joining_iso)
    for c in chars:
        forms = positional[c]
        for f in ("init", "medi", "fina"):
            if f in forms and forms[f] not in joining_all:
                joining_all.append(forms[f])

    lines: List[str] = []
    lines.append(f"@joining = [{' '.join(joining_all)}];")
    if medi_lhs:
        lines.append(f"@medi_lhs = [{' '.join(medi_lhs)}];")
        lines.append(f"@medi_rhs = [{' '.join(medi_rhs)}];")
    if fina_lhs:
        lines.append(f"@fina_lhs = [{' '.join(fina_lhs)}];")
        lines.append(f"@fina_rhs = [{' '.join(fina_rhs)}];")
    if init_lhs:
        lines.append(f"@init_lhs = [{' '.join(init_lhs)}];")
        lines.append(f"@init_rhs = [{' '.join(init_rhs)}];")
    lines.append("")

    # calt rules: most specific (medi: surrounded on both sides) first,
    # then fina (preceded only), then init (followed only). Anything left
    # over (no joining neighbour at all) stays as the iso form.
    lines.append("feature calt {")
    if medi_lhs:
        lines.append("    sub @joining @medi_lhs' @joining by @medi_rhs;")
    if fina_lhs:
        lines.append("    sub @joining @fina_lhs' by @fina_rhs;")
    if init_lhs:
        lines.append("    sub @init_lhs' @joining by @init_rhs;")
    lines.append("} calt;")

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
    form: str = "iso"        # iso, init, medi, fina — cursive positional
    # Per-glyph cursive connection x positions in submitted (post-xShift)
    # canvas coordinates. None falls back to the canvas-PAD assumption.
    entry_x: Optional[float] = None
    exit_x:  Optional[float] = None


# Canvas PAD value — the JS-side canvas adds this many pixels of padding on
# each side of the ink bbox. For cursive forms with connectors we strip this
# padding from the appropriate sides so consecutive letters actually touch.
CANVAS_PAD = 12


def build_otf(
    glyphs: List[GlyphData],
    font_name: str,
    font_style: str = "Regular",
    letter_spacing: int = 0,
    space_width: int = DEFAULT_ADVANCE_WIDTH,
    positional: Optional[Dict[str, Dict[str, str]]] = None,
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
            form=g.form,
            entry_x=g.entry_x, exit_x=g.exit_x,
        )
        # Keep CFF charstring width consistent with hmtx (both include letter_spacing)
        if letter_spacing != 0 and cs.program and isinstance(cs.program[0], (int, float)):
            cs.program[0] = advance + letter_spacing
        charstrings[g.glyph_name] = cs
        metrics[g.glyph_name] = (advance + letter_spacing, 0)

    # PostScript name must be unique per family+style combination.
    # fontTools' setupNameTable auto-computes nameID 6 as "family-style" for
    # non-Regular styles (e.g. "kjg-Line"); the CFF table's psName must match
    # exactly or strict validators (Windows Font Viewer) reject the font.
    if font_style == "Regular":
        ps_name = font_name
    else:
        ps_name = f"{font_name}-{font_style.replace(' ', '')}"

    fb.setupCFF(
        psName=ps_name,
        fontInfo={"version": "1.0", "FullName": f"{font_name} {font_style}",
                  "FamilyName": font_name, "Weight": font_style},
        charStringsDict=charstrings,
        privateDict=private_dict,
    )

    fb.setupHorizontalMetrics(metrics)

    fb.setupHorizontalHeader(ascent=ASCENDER, descent=DESCENDER)

    # nameID 6 (PostScript name) and nameID 4 (full name) are required for OTF.
    # fontTools' setupNameTable only writes records for keys it's given, so we
    # must pass these explicitly or strict validators (Windows Font Viewer)
    # reject the font as "not a valid font file".
    fb.setupNameTable({
        "familyName": font_name,
        "styleName": font_style,
        "psName": ps_name,
        "fullName": f"{font_name} {font_style}",
    })

    # Panose: 10-byte font classification. All-zero is "any" but some
    # validators (notably Windows) prefer at least bFamilyType to be set.
    from fontTools.ttLib.tables.O_S_2f_2 import Panose
    panose = Panose()
    panose.bFamilyType = 3  # Latin Hand Written

    fb.setupOS2(
        sTypoAscender=ASCENDER,
        sTypoDescender=DESCENDER,
        sTypoLineGap=0,
        usWinAscent=WIN_ASCENT,
        usWinDescent=WIN_DESCENT,
        sxHeight=X_HEIGHT,
        sCapHeight=CAP_HEIGHT,
        fsType=0,
        panose=panose,
        # fsSelection: 0x40 = REGULAR. fontTools defaults this to 0 which
        # Windows reads as "no style information" and treats as malformed.
        # USE_TYPO_METRICS (0x80) would also be useful but requires OS/2 v4+
        # while fontTools emits v3 by default — sticking with REGULAR alone.
        fsSelection=0x40,
    )

    # post table: italicAngle, underlinePosition, underlineThickness must
    # have sensible non-zero defaults — fontTools' setupPost() leaves them
    # at 0 which Windows treats as malformed.
    fb.setupPost(
        italicAngle=0.0,
        underlinePosition=-100,    # below baseline
        underlineThickness=50,
        isFixedPitch=0,
    )

    # head.created / head.modified default to 0 in fontTools, which becomes
    # 1904-01-01 in OpenType time — strict validators flag this as bogus.
    # Use the current time (seconds since 1904-01-01).
    import time
    mac_epoch_offset = 2082844800  # seconds between 1904-01-01 and 1970-01-01
    now = int(time.time()) + mac_epoch_offset
    fb.setupHead(unitsPerEm=UPM, created=now, modified=now)

    # Add OpenType features. Two independent contributions to `calt`:
    #   - alternates cycling (slot-based, print mode)
    #   - positional substitution (init/medi/fina, cursive mode)
    # Concatenated into one FEA blob; feaLib can compile multiple
    # `feature calt { ... }` blocks just fine.
    fea_warning = None
    fea_blocks: List[str] = []
    alternates = _collect_alternates(glyphs)
    if alternates:
        fea_blocks.append(_build_fea_code(alternates, all_glyph_names=glyph_order))
    if positional:
        fea_blocks.append(_build_cursive_fea_code(positional, available_glyphs=glyph_order))
    if fea_blocks:
        fea_code = "\n\n".join(b for b in fea_blocks if b)
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
    form: str = "iso",
    entry_x: Optional[float] = None,
    exit_x:  Optional[float] = None,
) -> Tuple["T2CharString", int]:
    """Build a CFF T2CharString by drawing SVG paths. Returns (charstring, advance_width)."""
    from fontTools.pens.t2CharStringPen import T2CharStringPen

    coord_scale = CELL_SCALE / upscale_factor if upscale_factor > 0 else CELL_SCALE
    if svg_width > 0:
        _, adv_extra = _bearing_offsets(form, coord_scale, svg_width, entry_x, exit_x)
        advance = int(svg_width * coord_scale) + adv_extra
    else:
        advance = DEFAULT_ADVANCE_WIDTH

    pen = T2CharStringPen(advance, glyphSet=None)

    _draw_svg_paths_to_pen(
        pen, svg_paths, svg_width, svg_height,
        baseline_y_in_svg, upscale_factor=upscale_factor, form=form,
        entry_x=entry_x, exit_x=exit_x,
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
