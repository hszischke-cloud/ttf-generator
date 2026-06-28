"""
font_builder.py — Assemble OTF font from vectorized glyphs using fonttools.

Coordinate system:
  - Source: SVG pixel space, y increases downward, origin top-left of upscaled glyph image
  - Target: Font UPM space (1000 units), y increases upward, baseline at y=0

Transform for each point:
  font_x = svg_x * scale_x + x_offset
  font_y = -(svg_y * scale_y) + y_offset   # negation = Y-axis flip
"""

import hashlib
import io
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from fontTools.fontBuilder import FontBuilder
from fontTools.feaLib.builder import addOpenTypeFeatures
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.colorLib.builder import buildCOLR, buildCPAL

from processing.perturb import perturb_glyph

# Canvas geometry (formerly TEMPLATE_SPEC). The drawing canvas uses these
# same guideline ratios so glyphs come in with consistent baselines.
CELL_HEIGHT_MM = 24.0
MM_TO_PX = 11.811                       # 300 dpi
GUIDELINE_TOP_RATIO       = 0.15        # CAP line
GUIDELINE_XHEIGHT_RATIO   = 0.42
GUIDELINE_BASELINE_RATIO  = 0.72
GUIDELINE_BOTTOM_RATIO    = 0.95        # descender (must match the UIs' GUIDELINES)

# Height of the drawing canvas in both UIs (DRAW_H). Every glyph arrives in
# this pixel space (svg_height=400, baseline_y=288).
CANVAS_HEIGHT_PX = 400

# ---------------------------------------------------------------------------
# Font metric constants (in UPM = 1000)
# ---------------------------------------------------------------------------

UPM = 1000
CAP_HEIGHT = 700        # nominal value CELL_SCALE is derived from — see below
DEFAULT_ADVANCE_WIDTH = 600

_cap_to_base  = GUIDELINE_BASELINE_RATIO - GUIDELINE_TOP_RATIO
_x_to_base    = GUIDELINE_BASELINE_RATIO - GUIDELINE_XHEIGHT_RATIO
_base_to_desc = GUIDELINE_BOTTOM_RATIO   - GUIDELINE_BASELINE_RATIO

# Cell-level scale: UPM per original canvas pixel (same for every glyph).
# Historically derived from the 24mm scan-template cell — kept verbatim so
# existing fonts keep their exact glyph scale on rebuild.
CELL_BASELINE_Y_PX = int(CELL_HEIGHT_MM * MM_TO_PX * _cap_to_base)
CELL_SCALE = CAP_HEIGHT / CELL_BASELINE_Y_PX

# Real ink landmarks in UPM. The 400px canvas is ~1.42× taller than the 24mm
# cell CELL_SCALE assumes, so ink drawn to the cap line actually lands at
# ~991 UPM and ink at the descender line at ~-400 UPM. Vertical metrics MUST
# be derived from this real canvas mapping: the old ratio×CAP_HEIGHT values
# (ascent 800/descent -221, win 900/200) understated the envelope, and
# renderers that clip to the declared metrics cut the bottoms off j/g/y
# descenders and the tops off cap-line capitals.
CAP_INK_UPM  = int(round(_cap_to_base  * CANVAS_HEIGHT_PX * CELL_SCALE))   # ≈ 991
X_HEIGHT     = int(round(_x_to_base    * CANVAS_HEIGHT_PX * CELL_SCALE))   # ≈ 522
DESC_INK_UPM = int(round(_base_to_desc * CANVAS_HEIGHT_PX * CELL_SCALE))   # ≈ 400

# Line-box metrics (hhea + OS/2 typo): cap/descender ink plus margin for the
# realistic pen's round caps and overshoot past the guides.
ASCENDER  = 1050
DESCENDER = -(DESC_INK_UPM + 50)        # ≈ -450

# Win metrics are CLIP bounds on Windows — cover the entire drawable canvas
# (y ∈ [0, 400] → ink ∈ [-487, +1252] plus stroke-cap overhang) so nothing
# is ever cut off, even strokes drawn past the guides.
WIN_ASCENT = 1280
WIN_DESCENT = 520


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

    # iso (print): no built-in buffer. The glyph sits flush at its ink edges so
    # advance == ink width and adjacent letters touch when letter_spacing is 0.
    # All spacing is added explicitly — globally via letter_spacing (tracking)
    # or per-glyph via side-bearing overrides. x_offset places the leftmost ink
    # (canvas x = PAD) at font x = 0; adv_extra strips the PAD on both sides.
    return -CANVAS_PAD * coord_scale, -int(round(2 * CANVAS_PAD * coord_scale))


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
    x_offset_override: Optional[float] = None,
) -> int:
    """
    Draw SVG paths into a fonttools pen, applying coordinate transform and
    cursive bearing adjustments per `form`.

    When `x_offset_override` is given it replaces the form-derived horizontal
    offset (used for manual left side-bearing adjustment on iso glyphs).

    Returns the advance width (in UPM units).
    """
    if not svg_paths or svg_width == 0 or svg_height == 0:
        return DEFAULT_ADVANCE_WIDTH

    coord_scale = CELL_SCALE / upscale_factor
    y_offset = baseline_y_in_svg * CELL_SCALE

    x_offset, adv_extra = _bearing_offsets(form, coord_scale, svg_width, entry_x, exit_x)
    if x_offset_override is not None:
        x_offset = x_offset_override
    advance_width = int(svg_width * coord_scale) + adv_extra

    pen.beginPath = getattr(pen, 'beginPath', None)

    # Round to integers once in font space so the forward and reverse passes of
    # each retrace share exactly the same integer lattice — prevents sub-pixel
    # slivers between the two passes when CFF quantises coordinates.
    def tx(sx: float) -> int:
        return round(sx * coord_scale + x_offset)

    def ty(sy: float) -> int:
        return round(-(sy * coord_scale) + y_offset)

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
                pen.moveTo((tx(sx), ty(sy)))
                current_x, current_y = sx, sy
                started = True

            elif cmd == 'L':
                sx, sy = args[0], args[1]
                pen.lineTo((tx(sx), ty(sy)))
                current_x, current_y = sx, sy

            elif cmd == 'C':
                for j in range(0, len(args), 6):
                    if j + 5 >= len(args):
                        break
                    x1, y1 = args[j], args[j + 1]
                    x2, y2 = args[j + 2], args[j + 3]
                    x, y = args[j + 4], args[j + 5]
                    pen.curveTo(
                        (tx(x1), ty(y1)),
                        (tx(x2), ty(y2)),
                        (tx(x),  ty(y)),
                    )
                    current_x, current_y = x, y

            elif cmd in ('Z', 'z'):
                if started:
                    pen.closePath()
                    started = False

        if started:
            pen.closePath()

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
    MAX_DIST = 4   # look back up to 4 glyphs for the previous occurrence.
                    # Each extra distance multiplies feaLib compile time by a
                    # large constant; 4 still covers typical syllable-scale
                    # repeats ("banana", "letter") while keeping the rule set
                    # small enough that font generation isn't dominated by
                    # feaLib compile time.

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
    # Per-glyph manual side bearings (UPM). When both are set on an iso glyph,
    # they override the auto bbox-derived advance and left bearing. lsb = gap
    # from origin to leftmost ink; rsb = gap from rightmost ink to advance.
    lsb_upm: Optional[int] = None
    rsb_upm: Optional[int] = None


# Canvas PAD value — the JS-side canvas adds this many pixels of padding on
# each side of the ink bbox. For cursive forms with connectors we strip this
# padding from the appropriate sides so consecutive letters actually touch.
CANVAS_PAD = 12


def default_bearings_upm(svg_width: int, upscale_factor: float = 1.0) -> Tuple[int, int]:
    """
    Default (lsb, rsb) in UPM for an iso glyph: (0, 0).

    The bearings are measured as space *beyond the ink edge*, so 0 means the
    side-bearing line sits exactly on the leftmost / rightmost ink point and
    adjacent letters touch. Inter-letter spacing is added explicitly via the
    global letter_spacing tracking or per-glyph overrides.
    """
    return 0, 0


def _override_offsets(
    lsb_upm: int, rsb_upm: int, svg_width: int, upscale_factor: float,
) -> Tuple[float, int]:
    """
    Translate explicit side bearings into (x_offset, advance) for an iso glyph.

    The glyph ink occupies [PAD, svg_width-PAD] px in submitted coords. To place
    the leftmost ink at font x = lsb_upm we shift the outline by x_offset; the
    advance then spans lsb + ink_width + rsb.
    """
    cs = CELL_SCALE / upscale_factor if upscale_factor > 0 else CELL_SCALE
    ink_w = max(0, svg_width - 2 * CANVAS_PAD) * cs
    x_offset = lsb_upm - CANVAS_PAD * cs
    advance = int(round(lsb_upm + ink_w + rsb_upm))
    return x_offset, max(0, advance)


def compute_glyph_advance(g: "GlyphData", letter_spacing: int = 0) -> int:
    """
    Return the final advance width (in UPM) for a single glyph, matching the
    formula used inside build_otf.  letter_spacing is added on top.
    """
    # Manual side-bearing override (iso/print glyphs only).
    if g.form == "iso" and g.lsb_upm is not None and g.rsb_upm is not None and g.svg_width > 0:
        _, adv = _override_offsets(g.lsb_upm, g.rsb_upm, g.svg_width, g.upscale_factor)
        return max(0, adv + letter_spacing)

    coord_scale = CELL_SCALE / g.upscale_factor if g.upscale_factor > 0 else CELL_SCALE
    if g.svg_width > 0:
        _, adv_extra = _bearing_offsets(g.form, coord_scale, g.svg_width, g.entry_x, g.exit_x)
        # Clamp to 0: very tight (negative) tracking must never produce a
        # negative advance, which is invalid in hmtx and reverses text flow.
        return max(0, int(g.svg_width * coord_scale) + adv_extra + letter_spacing)
    return max(0, DEFAULT_ADVANCE_WIDTH + letter_spacing)


def build_otf(
    glyphs: List[GlyphData],
    font_name: str,
    font_style: str = "Regular",
    letter_spacing: int = 0,
    space_width: int = DEFAULT_ADVANCE_WIDTH,
    positional: Optional[Dict[str, Dict[str, str]]] = None,
    perturb: bool = True,
    perturb_amplitude: float = 3.0,
    perturb_frequency: float = 0.13,
    forced_advances: Optional[Dict[str, int]] = None,
    color_layers: bool = True,
    base_color: Tuple[int, int, int] = (38, 32, 28),
    pool_color: Optional[Tuple[int, int, int]] = None,
    speck_color: Optional[Tuple[int, int, int]] = None,
    remove_overlaps: bool = False,
) -> bytes:
    """
    Assemble an OTF font from a list of vectorized glyphs.

    Args:
        glyphs: list of GlyphData, one per accepted glyph
        font_name: the font family name (user-supplied)
        font_style: style name (default "Regular")
        letter_spacing: extra UPM units added to every glyph's advance width (tracking)
        space_width: advance width for the space glyph in UPM units
        forced_advances: when provided, overrides the per-glyph advance width
            computation.  Keys are glyph names; "space" overrides the space
            advance.  Use this to guarantee that two fonts (e.g. regular and
            single-line) share exactly the same spacing.
        remove_overlaps: when True, each glyph's stroke contours are boolean-
            unioned into clean, non-overlapping outlines (see
            _remove_overlaps_to_pen).  Used for the dimensional/filled OTF so
            overlapping strokes don't leave white slivers; must stay False for
            the single-line plotter OTF, whose thin strokes are meant to overlap.

    Returns:
        Raw OTF bytes.
    """
    fb = FontBuilder(UPM, isTTF=False)

    # Build glyph order — base glyphs first; COLR layer glyphs (".pool",
    # ".speck") get appended inside the main loop and the order is committed
    # to FontBuilder once everything is collected.
    glyph_order = [".notdef", "space"]
    seen = set(glyph_order)
    for g in glyphs:
        if g.glyph_name not in seen:
            glyph_order.append(g.glyph_name)
            seen.add(g.glyph_name)

    # Character map (only primary glyphs / slot==0 map to Unicode). Layer
    # glyphs deliberately have no cmap entry — they're only reachable via COLR.
    cmap: Dict[int, str] = {0x0020: "space"}
    for g in glyphs:
        if g.slot == 0:
            cp = char_to_unicode(g.char)
            if cp is not None:
                cmap[cp] = g.glyph_name

    from fontTools.pens.t2CharStringPen import T2CharStringPen
    from fontTools.misc.psCharStrings import T2CharString

    private_dict = {"defaultWidthX": DEFAULT_ADVANCE_WIDTH, "nominalWidthX": 0}
    charstrings: Dict[str, any] = {}

    # Resolve space width: forced_advances["space"] wins if present.
    _space_width = (forced_advances.get("space", space_width)
                    if forced_advances else space_width)
    metrics: Dict[str, Tuple[int, int]] = {
        ".notdef": (500, 0),
        "space": (_space_width, 0),
    }

    charstrings[".notdef"] = _build_notdef_charstring()

    # space — encode the width explicitly as the first element of the CFF program.
    # CFF rule: if the charstring begins with a number before any drawing op, that number
    # is (advance - nominalWidthX). With nominalWidthX=0, encoding space_width directly
    # gives the correct advance. Using just ['endchar'] would fall back to defaultWidthX=600.
    from fontTools.misc.psCharStrings import T2CharString as _T2CS
    _space_cs = _T2CS()
    _space_cs.program = [_space_width, "endchar"]
    charstrings["space"] = _space_cs
    print(f"[font_builder] space_width={_space_width}  letter_spacing={letter_spacing}  program={_space_cs.program}")

    # Color layer info collected during the main loop, applied after setupCFF.
    # Maps each base glyph name to its (layer_glyph_name, palette_index) stack.
    color_glyph_layers: Dict[str, List[Tuple[str, int]]] = {}

    # COLR is built whenever color_layers is on. With perturb, the pool patches
    # add textured variation; without perturb (e.g. the line OTF), every glyph
    # still gets a single-layer COLR entry so renderers paint it in the user's
    # chosen palette colour instead of the document/CSS foreground colour.
    do_color = color_layers

    for g in glyphs:
        # Manual side-bearing override (iso/print glyphs only): shift the outline
        # by x_offset and fix the advance to lsb + ink + rsb.
        x_off_ov = adv_ov = None
        if (g.form == "iso" and g.lsb_upm is not None
                and g.rsb_upm is not None and g.svg_width > 0):
            x_off_ov, adv_ov = _override_offsets(
                g.lsb_upm, g.rsb_upm, g.svg_width, g.upscale_factor,
            )

        cs, advance, perturbed_contours = _build_charstring_from_svg(
            g.svg_paths, g.svg_width, g.svg_height,
            g.baseline_y_in_svg, g.is_lowercase,
            upscale_factor=g.upscale_factor,
            form=g.form,
            entry_x=g.entry_x, exit_x=g.exit_x,
            perturb=perturb,
            perturb_amplitude=perturb_amplitude,
            perturb_frequency=perturb_frequency,
            glyph_name=g.glyph_name,
            x_offset_override=x_off_ov,
            advance_override=adv_ov,
            remove_overlaps=remove_overlaps,
        )
        # Determine final advance: forced_advances wins, otherwise add letter_spacing.
        if forced_advances is not None and g.glyph_name in forced_advances:
            final_advance = forced_advances[g.glyph_name]
        else:
            # Clamp to 0 so extreme negative tracking can't yield a negative
            # advance width (invalid hmtx / reversed text).
            final_advance = max(0, advance + letter_spacing)
        # Keep CFF charstring width consistent with hmtx.
        if cs.program and isinstance(cs.program[0], (int, float)):
            cs.program[0] = final_advance
        charstrings[g.glyph_name] = cs
        metrics[g.glyph_name] = (final_advance, 0)

        if do_color:
            # Single-layer COLR entry painting the base outline in palette
            # index 0 (the user's chosen pen colour). Without this, glyphs
            # fall back to the cmap base in the renderer's foreground
            # colour (CSS `color`, document text colour) and ignore the
            # user's chosen pen colour entirely. No pool overlay layers
            # — the synthetic dot patches read as polka dots at preview
            # size and the canvas's per-stamp speed-whitening can't
            # round-trip through a vector OTF anyway. Divots stay (baked
            # into the base outline as half-octagonal CFF holes), which
            # gives natural-looking edge texture without interior spots.
            color_glyph_layers[g.glyph_name] = [(g.glyph_name, 0)]

    # Now that all glyphs (base + COLR layers) are known, commit them to the
    # FontBuilder. setupGlyphOrder/setupCharacterMap have to come before
    # setupCFF so the CFF table can index by glyph order.
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)

    # PostScript name (nameID 6): ASCII printable only, no spaces, max 63 chars.
    # The spec bans anything outside [A-Za-z0-9._-]; we collapse spaces to
    # hyphens and strip the rest so user-typed names never break opentype.js
    # or strict validators.
    import re as _re
    _ps_raw = font_name if font_style == "Regular" else f"{font_name}-{font_style}"
    ps_name = _re.sub(r'[^A-Za-z0-9._-]', '', _ps_raw.replace(' ', '-'))[:63]
    if not _re.search(r'[A-Za-z0-9]', ps_name):
        ps_name = "Untitled"

    fb.setupCFF(
        psName=ps_name,
        fontInfo={"version": "1.0", "FullName": f"{font_name} {font_style}",
                  "FamilyName": font_name, "Weight": font_style},
        charStringsDict=charstrings,
        privateDict=private_dict,
    )

    # CPAL palette — single entry, the user's chosen pen colour. With
    # no pool overlay layers there's nothing to put at indices 1+, so
    # the palette is just [base]. pool_color/speck_color params are
    # ignored now; kept on the signature for back-compat.
    # Divots are baked into the base outline as half-octagonal CFF holes
    # biting into the edge, so the background shows through them in
    # every renderer.
    if color_glyph_layers:
        br, bg, bb = base_color
        glyph_index_map = {name: i for i, name in enumerate(glyph_order)}
        try:
            colr_table = buildCOLR(
                color_glyph_layers, version=0, glyphMap=glyph_index_map,
            )
            cpal_table = buildCPAL([[
                (br / 255, bg / 255, bb / 255, 1.0),
            ]])
            fb.font['COLR'] = colr_table
            fb.font['CPAL'] = cpal_table
        except Exception as e:
            print(f"Warning: COLR/CPAL build failed (color layers skipped): {e}")

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
        sCapHeight=CAP_INK_UPM,
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
    return buf.getvalue(), fea_warning


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _CollectingPen:
    """
    Deferred pen that records all draw operations as plain tuples.

    After drawing, call `perturb_and_replay(real_pen, glyph_name, ...)` to
    apply organic micro-serrations and then replay the perturbed ops to a
    real fonttools pen.

    The `beginPath` attribute is a no-op slot required by
    `_draw_svg_paths_to_pen`, which sets it on the pen instance directly.
    """

    def __init__(self):
        self.beginPath = None        # compatibility shim
        self._contours: List[list] = []
        self._current: list = []

    def moveTo(self, pt):
        if self._current:
            self._contours.append(self._current)
        self._current = [('moveTo', pt)]

    def lineTo(self, pt):
        self._current.append(('lineTo', pt))

    def curveTo(self, *pts):
        # pts = ((cx1,cy1), (cx2,cy2), (x,y)) for cubic Bezier
        self._current.append(('curveTo', *pts))

    def closePath(self):
        self._current.append(('closePath',))
        self._contours.append(self._current)
        self._current = []

    def endPath(self):
        self._current.append(('endPath',))
        if self._current:
            self._contours.append(self._current)
        self._current = []

    def _flush(self):
        if self._current:
            self._contours.append(self._current)
            self._current = []

    @staticmethod
    def _replay_contour(contour: list, pen) -> None:
        for op in contour:
            name = op[0]
            if name == 'moveTo':
                pen.moveTo(op[1])
            elif name == 'lineTo':
                pen.lineTo(op[1])
            elif name == 'curveTo':
                pen.curveTo(*op[1:])
            elif name == 'closePath':
                pen.closePath()
            elif name == 'endPath':
                pen.endPath()

    def perturb_and_replay(
        self,
        pen,
        glyph_name: str,
        amplitude: float,
        frequency: float,
    ) -> None:
        """Perturb collected contours then replay them to *pen*.

        Stores the perturbed contours back on self so callers can reuse them
        (e.g. for COLR layer scatter generation) without re-perturbing.
        """
        self._flush()
        self._contours = perturb_glyph(
            self._contours, glyph_name,
            amplitude=amplitude, frequency=frequency,
        )
        for contour in self._contours:
            self._replay_contour(contour, pen)

    def replay(self, pen) -> None:
        """Replay without perturbation (passthrough mode)."""
        self._flush()
        for contour in self._contours:
            self._replay_contour(contour, pen)


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


def _remove_overlaps_to_pen(contours: List[list], out_pen) -> bool:
    """
    Boolean-union a glyph's stroke contours and replay the merged, non-overlapping
    outline to *out_pen*.

    Every realistic-ink stroke is emitted as its own closed contour, so a glyph's
    strokes overlap wherever the pen crossed its own path: the centre of an 'x',
    a 't' crossbar over the stem, the bowl of a 'b' meeting the stem, a loop that
    folds back on itself. Overlapping contours fill solid under the non-zero
    winding rule but leave white slivers under an even-odd rule, and overlaps in
    general trip up design/print tooling. Merging the strokes into clean,
    non-overlapping outlines makes the glyph render identically (and correctly)
    under both fill rules. Genuine counters — the holes in 'o', 'e', 'a', 'b', … —
    are preserved because each stroke keeps its own winding.

    Advance widths are computed elsewhere and are untouched, so spacing/bearings
    never shift.

    Returns True when the union was applied. On any failure — including
    skia-pathops not being installed — it replays the original contours unchanged
    and returns False, so the build never regresses relative to the raw outline.
    """
    try:
        import pathops
    except Exception:
        for c in contours:
            _CollectingPen._replay_contour(c, out_pen)
        return False

    def _feed(contour: list, pen) -> None:
        # Feed a contour to a pathops pen. A fill outline is always closed, so we
        # drop any closePath/endPath op and close explicitly at the end (pathops
        # boolean ops require closed contours).
        started = False
        for op in contour:
            kind = op[0]
            if kind == 'moveTo':
                pen.moveTo(op[1]); started = True
            elif kind == 'lineTo':
                pen.lineTo(op[1])
            elif kind == 'curveTo':
                pen.curveTo(*op[1:])
        if started:
            pen.closePath()

    try:
        operands = []
        for c in contours:
            sp = pathops.Path()
            _feed(c, sp.getPen())
            operands.append(sp)
        merged = pathops.Path()
        pathops.union(operands, merged.getPen())
        if any(True for _ in merged.contours):
            merged.draw(out_pen)
            return True
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"[font_builder] remove_overlaps failed ({exc}); using raw outline")

    for c in contours:
        _CollectingPen._replay_contour(c, out_pen)
    return False


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
    perturb: bool = False,
    perturb_amplitude: float = 3.0,
    perturb_frequency: float = 0.13,
    glyph_name: str = "",
    x_offset_override: Optional[float] = None,
    advance_override: Optional[int] = None,
    remove_overlaps: bool = False,
) -> Tuple["T2CharString", int, List[list]]:
    """
    Build a CFF T2CharString by drawing SVG paths.

    When *perturb* is True the contour points are passed through
    `_CollectingPen` → `perturb_glyph` before being committed to the
    charstring, adding organic edge micro-serrations.  The perturbation
    is applied in font UPM space (after the SVG→font coordinate transform)
    so the amplitude is consistent across all glyph sizes.

    Returns (charstring, advance_width, perturbed_contours). The third
    element is the list of pen-op contours (post-perturbation if applied),
    or an empty list when perturb=False — used by the COLR layer builder
    to scatter ink-variation patches inside the glyph.
    """
    from fontTools.pens.t2CharStringPen import T2CharStringPen

    coord_scale = CELL_SCALE / upscale_factor if upscale_factor > 0 else CELL_SCALE
    if advance_override is not None:
        advance = max(0, int(advance_override))
    elif svg_width > 0:
        _, adv_extra = _bearing_offsets(form, coord_scale, svg_width, entry_x, exit_x)
        advance = int(svg_width * coord_scale) + adv_extra
    else:
        advance = DEFAULT_ADVANCE_WIDTH

    real_pen = T2CharStringPen(advance, glyphSet=None)
    contours: List[list] = []

    if perturb and svg_paths:
        # Collect → perturb → carve divot holes → replay everything.
        # Divots used to live as a separate COLR layer painted near-white,
        # which left visible light dots on non-white backgrounds. Baking
        # them into the base outline as CFF holes makes the background
        # show through whatever colour it is, in every renderer (COLR-
        # aware or not).
        collector = _CollectingPen()
        _draw_svg_paths_to_pen(
            collector, svg_paths, svg_width, svg_height,
            baseline_y_in_svg, upscale_factor=upscale_factor, form=form,
            entry_x=entry_x, exit_x=exit_x, x_offset_override=x_offset_override,
        )
        collector._flush()
        perturbed = perturb_glyph(
            collector._contours, glyph_name or "unknown",
            amplitude=perturb_amplitude,
            frequency=perturb_frequency,
        )
        outer_sign = _detect_outer_sign(perturbed)
        hole_contours = _collect_speck_hole_contours(
            perturbed, glyph_name or "unknown", outer_sign,
        )
        for c in perturbed:
            _CollectingPen._replay_contour(c, real_pen)
        for c in hole_contours:
            _CollectingPen._replay_contour(c, real_pen)
        # Pool layers are computed from the base contours only — the holes
        # are already baked in, and we don't want pool patches placed on
        # the divot edges.
        contours = perturbed
    elif remove_overlaps:
        # Collect the stroke outlines, then boolean-union them so overlapping
        # strokes merge into clean, non-overlapping contours (no white slivers
        # under even-odd renderers, no overlap artifacts in print/design tools).
        collector = _CollectingPen()
        _draw_svg_paths_to_pen(
            collector, svg_paths, svg_width, svg_height,
            baseline_y_in_svg, upscale_factor=upscale_factor, form=form,
            entry_x=entry_x, exit_x=exit_x, x_offset_override=x_offset_override,
        )
        collector._flush()
        _remove_overlaps_to_pen(collector._contours, real_pen)
    else:
        _draw_svg_paths_to_pen(
            real_pen, svg_paths, svg_width, svg_height,
            baseline_y_in_svg, upscale_factor=upscale_factor, form=form,
            entry_x=entry_x, exit_x=exit_x, x_offset_override=x_offset_override,
        )

    return real_pen.getCharString(), advance, contours


# ---------------------------------------------------------------------------
# COLR/CPAL — ink-variation color layers
# ---------------------------------------------------------------------------

def _on_curve_xy(op: tuple) -> Tuple[float, float]:
    """Return the on-curve endpoint of a pen op (moveTo, lineTo, curveTo)."""
    return op[-1]  # last tuple element is always the on-curve point


def _signed_area(pts: List[Tuple[float, float]]) -> float:
    """Twice the signed polygon area. Positive ⇒ CCW, negative ⇒ CW. Used
    to pick the per-glyph outer winding so the outline-walk inset direction
    is correct for both the outer contour and any holes."""
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def _contours_bbox(contours: List[list]) -> Tuple[float, float, float, float]:
    """Bounding box of all on-curve endpoints across all contours."""
    xs: List[float] = []
    ys: List[float] = []
    for contour in contours:
        for op in contour:
            if op[0] in ('moveTo', 'lineTo', 'curveTo'):
                x, y = _on_curve_xy(op)
                xs.append(x); ys.append(y)
    if not xs:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _pip_single(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon. Returns True if (x,y) is strictly inside."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        yi, yj = poly[i][1], poly[j][1]
        if (yi > y) != (yj > y):
            xi, xj = poly[i][0], poly[j][0]
            x_int = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_int:
                inside = not inside
        j = i
    return inside


def _point_inside_contours(x: float, y: float, contours: List[list]) -> bool:
    """Even-odd containment across multiple contours — handles holes (e.g. 'o', 'a')."""
    inside = False
    for contour in contours:
        pts = [_on_curve_xy(op) for op in contour
               if op[0] in ('moveTo', 'lineTo', 'curveTo')]
        if _pip_single(x, y, pts):
            inside = not inside
    return inside


def _build_color_layer_charstring(
    contours: List[list],
    glyph_name: str,
    advance: int,
    kind: str,                              # 'pool', 'pool_light', or 'speck'
    outer_sign: float = -1.0,               # +1 for CCW-outer glyphs, -1 for CW
) -> Optional["T2CharString"]:
    """
    Walk every contour's on-curve points and drop small octagonal patches
    at noise-sampled positions.

    kinds
      'pool'       — denser-ink patches, inset inside the stroke. Only
                     fires when a secondary "darkness" noise is on the
                     dark half, so the dark/light split is mutually
                     exclusive with 'pool_light' at every candidate.
      'pool_light' — whitened-ink patches, inset inside the stroke. Fires
                     on the opposite half of the same darkness noise, so
                     pool + pool_light cover every candidate position
                     between them. Together they give the COLR overlay
                     visible ink-density variation that approximates the
                     canvas-side speed-whitening on outward bumps (which
                     can't otherwise round-trip through a vector OTF).
      'speck'      — edge-straddling light divots. Historical; the
                     production path now bakes divots into the base
                     outline as CFF holes (see _collect_speck_hole_contours)
                     so they show the actual background instead of a
                     fixed light tint.

    *outer_sign* picks the inward normal direction once per glyph based
    on the largest contour's winding — without this, holes in 'o', 'a',
    'B' etc. would inset the wrong way.
    """
    if not contours:
        return None

    # Pool and pool_light share a single seed so they generate the same
    # candidate set; the split below decides which one each position lands in.
    seed_kind = 'pool' if kind in ('pool', 'pool_light') else kind
    seed = int(hashlib.md5((glyph_name + seed_kind).encode()).hexdigest()[:6], 16)
    # Independent seed for the darkness split so it isn't correlated with
    # the patch-placement noise.
    split_seed = int(hashlib.md5((glyph_name + 'pool_split').encode()).hexdigest()[:6], 16)

    # Outline-walking parameters (font UPM space)
    noise_freq = 0.06           # ~16 UPM per noise cycle
    # Higher threshold than before drops patch density roughly in half.
    # At display sizes this stops the patches from reading as a regular
    # polka-dot pattern marching along every stroke.
    threshold = 0.45
    # Patches sit closer to the original 4.5–7.5 UPM range. The previous
    # 10–18 UPM bump made each one read as a discrete dot at display
    # sizes (which is how the user sees the preview) instead of blending
    # into ink texture. Anti-aliasing carries the smaller patches as
    # subtle tone variation at preview size and clear-but-not-spotty
    # texture at very large sizes.
    radius_min, radius_max = 5.0, 8.0
    inset_frac = 0.7 if kind in ('pool', 'pool_light') else 0.0
    steps = 8                   # octagonal patches

    # Darkness-split noise sampled at a different frequency so it doesn't
    # mirror the placement noise (which would line dark/light patches up
    # with stroke curvature instead of looking randomly distributed).
    split_freq = 0.037

    pen = T2CharStringPen(advance, glyphSet=None)
    drew_any = False

    for c_idx, contour in enumerate(contours):
        on_pts = [op[-1] for op in contour
                  if op[0] in ('moveTo', 'lineTo', 'curveTo')]
        n = len(on_pts)
        if n < 4:
            continue
        closed = any(op[0] == 'closePath' for op in contour)

        # Cumulative arc length along the contour's on-curve points
        arc = [0.0]
        for i in range(1, n):
            dx = on_pts[i][0] - on_pts[i - 1][0]
            dy = on_pts[i][1] - on_pts[i - 1][1]
            arc.append(arc[-1] + math.hypot(dx, dy))
        if arc[-1] < 30:
            continue

        # Per-contour phase keeps adjacent contours uncorrelated
        phase_kind_offset = 0.0 if kind in ('pool', 'pool_light') else 50.3
        phase = (c_idx * 19.3 + seed * 0.137 + phase_kind_offset) % 1024
        split_phase = (c_idx * 11.7 + split_seed * 0.211) % 1024

        for i in range(n):
            # Local tangent (centered difference, wraps on closed contours)
            if closed:
                prev_pt = on_pts[(i - 1) % n]
                next_pt = on_pts[(i + 1) % n]
            else:
                prev_pt = on_pts[max(0, i - 1)]
                next_pt = on_pts[min(n - 1, i + 1)]
            dx = next_pt[0] - prev_pt[0]
            dy = next_pt[1] - prev_pt[1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            tx, ty = dx / length, dy / length
            inward_x = -ty * outer_sign
            inward_y =  tx * outer_sign

            nz = _py_noise1d(arc[i] * noise_freq + phase)
            if kind in ('pool', 'pool_light'):
                if nz <= threshold:
                    continue
                magnitude = nz - threshold
                # Darkness split: positive split-noise → pool_light,
                # negative → pool. Same threshold of 0 keeps the two halves
                # roughly equal in count.
                split_nz = _py_noise1d(arc[i] * split_freq + split_phase)
                is_light = split_nz > 0.0
                if kind == 'pool' and is_light:
                    continue
                if kind == 'pool_light' and not is_light:
                    continue
            else:
                if nz >= -threshold:
                    continue
                magnitude = -nz - threshold
            intensity = min(1.0, magnitude / (1.0 - threshold))
            radius = radius_min + intensity * (radius_max - radius_min)

            cx = on_pts[i][0] + inward_x * radius * inset_frac
            cy = on_pts[i][1] + inward_y * radius * inset_frac

            for k in range(steps):
                ang = (k / steps) * 2 * math.pi
                xx = cx + math.cos(ang) * radius
                yy = cy + math.sin(ang) * radius
                if k == 0:
                    pen.moveTo((round(xx), round(yy)))
                else:
                    pen.lineTo((round(xx), round(yy)))
            pen.closePath()
            drew_any = True

    return pen.getCharString() if drew_any else None


def _detect_outer_sign(contours: List[list]) -> float:
    """Determine outer-contour winding direction from the largest contour.
    Returns +1 for CCW outer, -1 for CW outer (this pipeline's default).
    Used so divot holes and patch insets always go the right way for the
    glyph at hand — 'o', 'a', 'B' etc. otherwise place patches outside the
    ink."""
    largest_area = 0.0
    outer_sign = -1.0
    for contour in contours:
        on_pts = [op[-1] for op in contour
                  if op[0] in ('moveTo', 'lineTo', 'curveTo')]
        if len(on_pts) < 3:
            continue
        area = _signed_area(on_pts)
        if abs(area) > largest_area:
            largest_area = abs(area)
            outer_sign = 1.0 if area > 0 else -1.0
    return outer_sign


def _collect_speck_hole_contours(
    contours: List[list],
    glyph_name: str,
    outer_sign: float,
) -> List[list]:
    """
    Build divot patches as half-octagonal contours sitting flush on the
    stroke edge, with the curved side biting inward — exactly the
    half-moon "edge eaten by paper" effect the canvas pen draws with its
    polygon-clipped white discs.

    Half-octagons sit entirely on the inside-half of the local edge, so
    nothing pokes outside the base outline (which would otherwise add
    ink under CFF's non-zero winding rule and produce ugly little
    bumps next to every divot).

    Winding is chosen at construction time so the shape acts as a hole
    against an outer contour with sign *outer_sign*.

    Returned contours are sequences of pen ops (moveTo / lineTo /
    closePath), ready to feed back to a T2CharStringPen.
    """
    if not contours:
        return []

    seed = int(hashlib.md5((glyph_name + 'speck').encode()).hexdigest()[:6], 16)

    noise_freq = 0.06
    threshold = 0.25
    # Half-octagons bite only as deep as their radius (vs a full octagon
    # which could pierce all the way through a thin stroke) so we can run
    # them a touch bigger than the original 3.5–6 UPM speck range. 5–8
    # UPM matches the pool patches' scale so divots read as edge texture
    # rather than discrete oversized bites.
    radius_min, radius_max = 5.0, 8.0
    steps = 8

    holes: List[list] = []

    for c_idx, contour in enumerate(contours):
        on_pts = [op[-1] for op in contour
                  if op[0] in ('moveTo', 'lineTo', 'curveTo')]
        n = len(on_pts)
        if n < 4:
            continue
        closed = any(op[0] == 'closePath' for op in contour)

        arc = [0.0]
        for i in range(1, n):
            dx = on_pts[i][0] - on_pts[i - 1][0]
            dy = on_pts[i][1] - on_pts[i - 1][1]
            arc.append(arc[-1] + math.hypot(dx, dy))
        if arc[-1] < 30:
            continue

        phase = (c_idx * 19.3 + seed * 0.137 + 50.3) % 1024

        for i in range(n):
            if closed:
                prev_pt = on_pts[(i - 1) % n]
                next_pt = on_pts[(i + 1) % n]
            else:
                prev_pt = on_pts[max(0, i - 1)]
                next_pt = on_pts[min(n - 1, i + 1)]
            dx = next_pt[0] - prev_pt[0]
            dy = next_pt[1] - prev_pt[1]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            tx, ty = dx / length, dy / length
            inward_x = -ty * outer_sign
            inward_y =  tx * outer_sign

            nz = _py_noise1d(arc[i] * noise_freq + phase)
            if nz >= -threshold:
                continue
            magnitude = -nz - threshold
            intensity = min(1.0, magnitude / (1.0 - threshold))
            radius = radius_min + intensity * (radius_max - radius_min)

            px, py = on_pts[i]

            # Half-octagon vertices: two endpoints sit on the edge tangent
            # at ±radius from the centre, three arc vertices bulge inward
            # along the normal. The shape is entirely on the inside-half
            # of the edge so nothing pokes outside the base outline.
            #
            # Edge-right → inward-right → apex → inward-left → edge-left,
            # close back along the diameter (implicit via closePath).
            inv_sqrt2 = 0.70710678
            pts = [
                (px + radius * tx,                                          py + radius * ty),
                (px + inv_sqrt2 * radius * tx + inv_sqrt2 * radius * inward_x,
                 py + inv_sqrt2 * radius * ty + inv_sqrt2 * radius * inward_y),
                (px + radius * inward_x,                                    py + radius * inward_y),
                (px - inv_sqrt2 * radius * tx + inv_sqrt2 * radius * inward_x,
                 py - inv_sqrt2 * radius * ty + inv_sqrt2 * radius * inward_y),
                (px - radius * tx,                                          py - radius * ty),
            ]

            # Winding sanity: the hole must wind OPPOSITE to the outer
            # contour. _signed_area > 0 ⇒ CCW; outer_sign = -1 means CW
            # outer (negative area), so we want positive area here.
            # If construction came out same-sign as the outer (rare —
            # depends on inward_x/y orientation), reverse the vertex order.
            sa = _signed_area(pts)
            if outer_sign * sa > 0:
                pts = list(reversed(pts))

            ops: list = [('moveTo', (round(pts[0][0]), round(pts[0][1])))]
            for p in pts[1:]:
                ops.append(('lineTo', (round(p[0]), round(p[1]))))
            ops.append(('closePath',))
            holes.append(ops)

    return holes


def _py_hash1d(i: int) -> float:
    """1D integer hash returning a float in [-1, +1]. Matches the canvas-side
    `_inkHash` closely enough that the two halves of the pipeline produce
    visually consistent variation patterns."""
    h = (i ^ 0x27d4eb2d) & 0xffffffff
    h = (h * 0x27d4eb2d) & 0xffffffff
    h ^= (h >> 15)
    return ((h & 0xffff) / 32767.5) - 1.0


def _py_noise1d(x: float) -> float:
    """Smoothstep-interpolated 1D value noise in [-1, +1]."""
    i = math.floor(x)
    f = x - i
    a = _py_hash1d(int(i))
    b = _py_hash1d(int(i) + 1)
    t = f * f * (3 - 2 * f)
    return a + t * (b - a)


def _build_notdef_charstring() -> "T2CharString":
    """Simple box glyph for .notdef — visible placeholder for missing characters."""
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    pen = T2CharStringPen(500, glyphSet=None)
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((450, 700))
    pen.lineTo((450, 0))
    pen.closePath()
    return pen.getCharString()


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
