"""
proof_sheet.py — Render a built OTF into an SVG "contact sheet" for visual QA.

The single-line (Line) font is built from raw pen centerlines and warns that it
"may not render correctly in browsers". This module reads the *actual built OTF*
back, extracts each glyph's literal contours, and lays them out in a labelled
grid so the user can eyeball that every glyph survived the CFF round-trip.

Why render the outlines directly (instead of an @font-face HTML sheet): the line
font's failure modes are collapsed strokes, wrong winding, and dropped glyphs.
A browser @font-face render hides exactly those bugs; drawing the contours the
font actually stores surfaces them. Metric guides (baseline / x-height / cap /
descender) and an advance-width marker are drawn per cell, and degenerate glyphs
(tiny or empty outlines) get a warning border.

Pure fonttools — no new dependencies.
"""

from typing import Dict, List, Optional, Tuple

from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.boundsPen import ControlBoundsPen

from processing.font_builder import (
    ASCENDER, CAP_INK_UPM, CHAR_TO_GLYPH_NAME, DESCENDER, UPM, X_HEIGHT,
)

# glyph base name -> codepoint, so digit/punctuation names (e.g. "zero",
# "period") can be categorised and ordered even when not present in cmap.
_NAME_TO_CP: Dict[str, int] = {name: ord(ch) for ch, name in CHAR_TO_GLYPH_NAME.items()}

# Cap guide for the proof: where cap-line ink actually lands (real canvas
# mapping), not the nominal CAP_HEIGHT the scale is derived from.
CAP_HEIGHT = CAP_INK_UPM

# Cell geometry (px in the output SVG).
SCALE = 0.14                      # px per font unit
COLS = 8
PAD_X = 18                        # left inset of the glyph within its cell
PAD_TOP = 16                      # space above the ascender line
LABEL_H = 34                      # label strip beneath the glyph area
GLYPH_H = int((ASCENDER - DESCENDER) * SCALE)   # height of the metric band
CELL_W = 150
CELL_H = PAD_TOP + GLYPH_H + LABEL_H
HEADER_H = 96

# A glyph whose control-box is smaller than this (font units) on either axis is
# almost certainly a degenerate sliver — the classic line-font round-trip bug.
MIN_DIM_UNITS = 12


def _reverse_cmap(font: TTFont) -> Dict[str, int]:
    """glyph name -> Unicode codepoint (best cmap)."""
    out: Dict[str, int] = {}
    try:
        for cp, name in font.getBestCmap().items():
            out.setdefault(name, cp)
    except Exception:
        pass
    return out


def _base_codepoint(name: str, rev_cmap: Dict[str, int]) -> Optional[int]:
    """
    Codepoint of a glyph's *base* character (the part before any '.suffix').

    Variants (a.alt1, a.init, …) aren't in cmap, so we resolve via the base
    name: the cmap first, then a single-char name, then the builder's
    char→name table reversed. Returns None for genuinely unmappable bases.
    """
    base = name.split(".", 1)[0]
    cp = rev_cmap.get(base)
    if cp is not None:
        return cp
    if len(base) == 1:
        return ord(base)
    return _NAME_TO_CP.get(base)


def _variant_rank(name: str) -> Tuple[int, str]:
    """Ordering among one base's forms: base itself, then alt1, alt2…, then
    init/medi/fina. Keeps each alternate directly after the glyph it varies."""
    if "." not in name:
        return (0, "")
    suffix = name.split(".", 1)[1]
    if suffix.startswith("alt"):
        try:
            return (1, f"{int(suffix[3:]):04d}")
        except ValueError:
            return (1, suffix)
    form_order = {"init": 0, "medi": 1, "fina": 2}
    return (2, f"{form_order.get(suffix, 9)}_{suffix}")


def _category(cp: Optional[int]) -> int:
    """Sort bucket: 0 uppercase, 1 lowercase, 2 digits, 3 other, 4 unmappable."""
    if cp is None:
        return 4
    c = chr(cp)
    if c.isalpha() and c.isupper():
        return 0
    if c.isalpha() and c.islower():
        return 1
    if c.isdigit():
        return 2
    return 3


def _ordered_glyph_names(font: TTFont, rev_cmap: Dict[str, int]) -> List[str]:
    """
    Glyph order for the proof sheet: uppercase, then lowercase, then digits,
    then special characters. Within each group glyphs sort by codepoint, and
    every alternate/positional form appears immediately after its base glyph.
    """
    names = [n for n in font.getGlyphOrder() if n not in (".notdef", "space")]

    def sort_key(n: str) -> Tuple[int, int, Tuple[int, str]]:
        cp = _base_codepoint(n, rev_cmap)
        cp_sort = cp if cp is not None else 0x10FFFF
        return (_category(cp), cp_sort, _variant_rank(n))

    return sorted(names, key=sort_key)


def _label_for(name: str, rev_cmap: Dict[str, int]) -> str:
    """Human-readable cell label: the character if mapped, plus the glyph name."""
    cp = rev_cmap.get(name)
    if cp is not None and 0x21 <= cp <= 0x7E:
        return f"{chr(cp)}  {name}"
    return name


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def render_proof_svg(
    otf_bytes: bytes,
    title: str,
    is_line: bool,
    skipped: Optional[List[str]] = None,
) -> str:
    """
    Render `otf_bytes` into a self-contained SVG contact sheet.

    Args:
        otf_bytes: the built OTF binary.
        title:     heading text (e.g. "MyFont — Line").
        is_line:   True for the centerline font — outlines are also stroked so
                   the hairline shape is visible at cell size.
        skipped:   glyph ids the build dropped from the line font (rendered as
                   explicit MISSING cells so the user knows they're absent).
    """
    font = TTFont(io_bytes(otf_bytes))
    glyph_set = font.getGlyphSet()
    hmtx = font["hmtx"]
    rev_cmap = _reverse_cmap(font)
    names = _ordered_glyph_names(font, rev_cmap)

    skipped = skipped or []
    cells: List[str] = []

    # Real glyph cells.
    for name in names:
        try:
            adv = hmtx[name][0]
        except Exception:
            adv = 0
        cells.append(_render_cell(name, glyph_set, adv, is_line, rev_cmap))

    # Explicit MISSING cells for glyphs the line build dropped entirely.
    for gid in skipped:
        cells.append(_render_missing_cell(gid))

    if not cells:
        cells.append(_render_missing_cell("(no glyphs)"))

    rows = (len(cells) + COLS - 1) // COLS
    width = COLS * CELL_W
    height = HEADER_H + rows * CELL_H

    parts: List[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="system-ui, sans-serif">'
    )
    parts.append(f'<rect width="{width}" height="{height}" fill="#ffffff"/>')

    # Header + legend.
    parts.append(
        f'<text x="20" y="38" font-size="22" font-weight="600" '
        f'fill="#26201c">{_esc(title)}</text>'
    )
    parts.append(
        f'<text x="20" y="62" font-size="13" fill="#6b5d50">'
        f'{len(names)} glyphs rendered from the actual OTF outlines · '
        f'guides: baseline / x-height / cap / descender · '
        f'red border = degenerate (tiny/empty) outline'
        f'{f" · {len(skipped)} skipped" if skipped else ""}</text>'
    )
    parts.append(
        f'<line x1="20" y1="{HEADER_H-12}" x2="{width-20}" y2="{HEADER_H-12}" '
        f'stroke="#e3dad0" stroke-width="1"/>'
    )

    # Grid.
    for i, cell in enumerate(cells):
        cx = (i % COLS) * CELL_W
        cy = HEADER_H + (i // COLS) * CELL_H
        parts.append(f'<g transform="translate({cx},{cy})">{cell}</g>')

    parts.append("</svg>")
    return "".join(parts)


def _render_cell(
    name: str,
    glyph_set,
    advance: int,
    is_line: bool,
    rev_cmap: Dict[str, int],
) -> str:
    """One grid cell: metric guides + the glyph outline + label."""
    # Control bounds tell us whether the outline is real or degenerate.
    bp = ControlBoundsPen(glyph_set)
    try:
        glyph_set[name].draw(bp)
    except Exception:
        bp.bounds = None
    bounds = bp.bounds

    degenerate = False
    if bounds is None:
        degenerate = True
    else:
        xmin, ymin, xmax, ymax = bounds
        if (xmax - xmin) < MIN_DIM_UNITS or (ymax - ymin) < MIN_DIM_UNITS:
            degenerate = True

    # Extract the outline path (font units, y-up).
    sp = SVGPathPen(glyph_set)
    try:
        glyph_set[name].draw(sp)
        d = sp.getCommands()
    except Exception:
        d = ""

    inner_w = CELL_W - 1
    inner_h = CELL_H - 1
    baseline_y = PAD_TOP + ASCENDER * SCALE   # cell-px y of the baseline

    def gy(units: float) -> float:
        return baseline_y - units * SCALE

    border = "#d8453a" if degenerate else "#eee6da"
    border_w = 2 if degenerate else 1

    out: List[str] = []
    out.append(
        f'<rect x="0.5" y="0.5" width="{inner_w}" height="{inner_h}" '
        f'fill="#fbfaf7" stroke="{border}" stroke-width="{border_w}"/>'
    )

    # Metric guides.
    guides = [
        (CAP_HEIGHT, "#cfe0d8"),
        (X_HEIGHT, "#cfe0d8"),
        (0, "#9bbcaf"),          # baseline — emphasised
        (DESCENDER, "#cfe0d8"),
    ]
    for val, col in guides:
        y = gy(val)
        sw = 1.4 if val == 0 else 0.8
        out.append(
            f'<line x1="{PAD_X}" y1="{y:.1f}" x2="{CELL_W-6}" y2="{y:.1f}" '
            f'stroke="{col}" stroke-width="{sw}"/>'
        )

    # Advance-width marker (left edge at x=0, right edge at advance).
    if advance > 0:
        ax = PAD_X + advance * SCALE
        out.append(
            f'<line x1="{ax:.1f}" y1="{gy(ASCENDER):.1f}" x2="{ax:.1f}" '
            f'y2="{gy(DESCENDER):.1f}" stroke="#e7c9a0" stroke-width="0.8" '
            f'stroke-dasharray="2 2"/>'
        )

    # The glyph itself: translate to baseline origin, flip y, scale. Path data is
    # in font units so stroke-width below is also in font units.
    if d:
        if is_line:
            # Fill + stroke so the hairline centerline reads at cell size.
            style = 'fill="#1a1a1a" stroke="#1a1a1a" stroke-width="10"'
        else:
            style = 'fill="#1a1a1a"'
        out.append(
            f'<g transform="translate({PAD_X},{baseline_y:.1f}) '
            f'scale({SCALE},{-SCALE})">'
            f'<path d="{d}" {style}/></g>'
        )

    # Label strip.
    label = _label_for(name, rev_cmap)
    ly = CELL_H - 12
    out.append(
        f'<text x="{CELL_W/2:.0f}" y="{ly}" font-size="12" text-anchor="middle" '
        f'fill="#5a4f44">{_esc(label)}</text>'
    )
    if degenerate:
        out.append(
            f'<text x="{CELL_W/2:.0f}" y="{ly-15}" font-size="10" '
            f'text-anchor="middle" fill="#d8453a">⚠ check outline</text>'
        )

    return "".join(out)


def _render_missing_cell(gid: str) -> str:
    inner_w = CELL_W - 1
    inner_h = CELL_H - 1
    return (
        f'<rect x="0.5" y="0.5" width="{inner_w}" height="{inner_h}" '
        f'fill="#fdf3f2" stroke="#d8453a" stroke-width="1" '
        f'stroke-dasharray="4 3"/>'
        f'<text x="{CELL_W/2:.0f}" y="{CELL_H/2:.0f}" font-size="13" '
        f'text-anchor="middle" fill="#d8453a">MISSING</text>'
        f'<text x="{CELL_W/2:.0f}" y="{CELL_H-12}" font-size="12" '
        f'text-anchor="middle" fill="#5a4f44">{_esc(gid)}</text>'
    )


def io_bytes(b: bytes):
    """Tiny local helper so TTFont can read from an in-memory buffer."""
    import io
    return io.BytesIO(b)
