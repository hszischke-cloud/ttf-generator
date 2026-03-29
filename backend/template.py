"""
template.py — PDF template generation for the handwriting font collector.

TEMPLATE_SPEC is the single source of truth for all cell positions.
It is imported by processing/alignment.py and processing/extraction.py.
"""

import io
from dataclasses import dataclass, field
from typing import List, Tuple

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas as rl_canvas

# ---------------------------------------------------------------------------
# TEMPLATE_SPEC — all measurements in mm, shared with extraction pipeline
# ---------------------------------------------------------------------------

MM_TO_PT = 72 / 25.4  # reportlab works in points

TEMPLATE_SPEC = {
    # Page
    "page_width_mm": 210,
    "page_height_mm": 297,
    # Margins (from page edge to first cell)
    "margin_left_mm": 12,
    "margin_top_mm": 20,
    # Cell dimensions
    "cell_width_mm": 18,
    "cell_height_mm": 24,
    "cell_cols": 10,
    # Cell interior guide lines (as fraction of cell height from top)
    "guideline_top_ratio": 0.15,        # cap height (top of uppercase letters)
    "guideline_xheight_ratio": 0.42,    # x-height (top of lowercase letters like a, e, x)
    "guideline_baseline_ratio": 0.72,   # baseline — write all letters sitting on this line
    "guideline_bottom_ratio": 0.90,     # descender line (bottom of g, p, y etc.)
    # Left gutter inside each cell — x-height and baseline reference marks live here.
    # The inner writing box starts at cell_x + guide_gutter_mm. Extraction crops
    # to the inner box, so gutter content is never seen by the detector.
    "guide_gutter_mm": 3.5,
    # Registration markers
    "marker_diameter_mm": 8,
    "marker_padding_mm": 4,
    # Characters per page
    # Page 0: A-Z then a-z (52 glyphs)
    # Page 1: 0-9, punctuation, then alternate cells
    "alt_chars": list("etaoinshrd"),   # top-10 most common — get 2 extra cells each
    "alt_slots": 2,                    # number of additional alternate glyphs per char
}

# Derived pixel constant used by extraction pipeline (300 dpi)
DPI = 300
MM_TO_PX = DPI / 25.4  # 1 mm = 11.811 px at 300 dpi

# Character sets
UPPERCASE = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LOWERCASE = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
PUNCTUATION = list(".,;:!?\"'()-/@&#")  # 16 chars

# Glyph order: (character, slot_index) where slot 0 = primary, 1/2 = alternates
def build_glyph_order() -> List[Tuple[str, int]]:
    order = []
    alt_chars = set(TEMPLATE_SPEC["alt_chars"])
    alt_slots = TEMPLATE_SPEC["alt_slots"]

    # Page 0: uppercase + lowercase primaries
    for ch in UPPERCASE + LOWERCASE:
        order.append((ch, 0))

    # Page 1: digits + punctuation primaries
    for ch in DIGITS + PUNCTUATION:
        order.append((ch, 0))

    # Alternates section
    for ch in TEMPLATE_SPEC["alt_chars"]:
        for slot in range(1, alt_slots + 1):
            order.append((ch, slot))

    return order

GLYPH_ORDER = build_glyph_order()

# ---------------------------------------------------------------------------
# Cell layout computation (in mm, top-left origin)
# ---------------------------------------------------------------------------

@dataclass
class CellDef:
    char: str
    slot: int        # 0 = primary, 1+ = alternate
    page: int
    col: int
    row: int

    @property
    def x_mm(self) -> float:
        return (TEMPLATE_SPEC["margin_left_mm"]
                + self.col * TEMPLATE_SPEC["cell_width_mm"])

    @property
    def y_mm(self) -> float:
        return (TEMPLATE_SPEC["margin_top_mm"]
                + self.row * TEMPLATE_SPEC["cell_height_mm"])


def compute_cell_layout() -> List[CellDef]:
    """Return all cell definitions across both pages."""
    cells: List[CellDef] = []
    cols = TEMPLATE_SPEC["cell_cols"]
    alt_chars = set(TEMPLATE_SPEC["alt_chars"])

    # --- Page 0: A-Z then a-z ---
    page0_chars = [(ch, 0) for ch in UPPERCASE + LOWERCASE]
    for idx, (ch, slot) in enumerate(page0_chars):
        col = idx % cols
        row = idx // cols
        cells.append(CellDef(char=ch, slot=slot, page=0, col=col, row=row))

    # --- Page 1: digits, punctuation ---
    page1_primary = [(ch, 0) for ch in DIGITS + PUNCTUATION]
    for idx, (ch, slot) in enumerate(page1_primary):
        col = idx % cols
        row = idx // cols
        cells.append(CellDef(char=ch, slot=slot, page=1, col=col, row=row))

    # Alternates on page 1, below the primary section
    p1_primary_rows = (len(page1_primary) + cols - 1) // cols
    alt_slots = TEMPLATE_SPEC["alt_slots"]
    alt_list = []
    for ch in TEMPLATE_SPEC["alt_chars"]:
        for slot in range(1, alt_slots + 1):
            alt_list.append((ch, slot))

    for idx, (ch, slot) in enumerate(alt_list):
        row_offset = p1_primary_rows + 1  # one blank row gap
        col = idx % cols
        row = row_offset + idx // cols
        cells.append(CellDef(char=ch, slot=slot, page=1, col=col, row=row))

    return cells

CELL_LAYOUT = compute_cell_layout()

# ---------------------------------------------------------------------------
# PDF Generation
# ---------------------------------------------------------------------------

def _draw_registration_marker(c: rl_canvas.Canvas, x_pt: float, y_pt: float):
    """Draw a filled concentric-circle registration marker in pt coordinates."""
    pad = TEMPLATE_SPEC["marker_padding_mm"] * MM_TO_PT
    diam = TEMPLATE_SPEC["marker_diameter_mm"] * MM_TO_PT
    r_outer = diam / 2
    r_inner = r_outer * 0.45

    c.setFillColor(colors.black)
    c.circle(x_pt, y_pt, r_outer, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.circle(x_pt, y_pt, r_inner, fill=1, stroke=0)
    c.setFillColor(colors.black)
    c.circle(x_pt, y_pt, r_inner * 0.4, fill=1, stroke=0)


def _draw_page(c: rl_canvas.Canvas, page_num: int, cells_on_page: List[CellDef]):
    pw_pt = TEMPLATE_SPEC["page_width_mm"] * MM_TO_PT
    ph_pt = TEMPLATE_SPEC["page_height_mm"] * MM_TO_PT
    cw_pt = TEMPLATE_SPEC["cell_width_mm"] * MM_TO_PT
    ch_pt = TEMPLATE_SPEC["cell_height_mm"] * MM_TO_PT
    pad_mm = TEMPLATE_SPEC["marker_padding_mm"]
    diam_mm = TEMPLATE_SPEC["marker_diameter_mm"]

    # Header
    c.setFont("Helvetica-Bold", 10)
    c.setFillColor(colors.black)
    c.drawCentredString(pw_pt / 2, ph_pt - 10 * MM_TO_PT, "Handwriting Font Template")
    c.setFont("Helvetica", 6)
    c.drawCentredString(
        pw_pt / 2, ph_pt - 14 * MM_TO_PT,
        "Lines from top: CAP (top of uppercase), x (top of lowercase), base (sit letters here), desc (descenders). Keep all 4 markers visible."
    )

    # Registration markers (at corners)
    marker_r = (diam_mm / 2 + pad_mm) * MM_TO_PT
    marker_positions = [
        (marker_r, ph_pt - marker_r),               # top-left
        (pw_pt - marker_r, ph_pt - marker_r),        # top-right
        (marker_r, marker_r),                         # bottom-left
        (pw_pt - marker_r, marker_r),                 # bottom-right
    ]
    for mx, my in marker_positions:
        _draw_registration_marker(c, mx, my)

    # Alternates section header (page 1 only)
    if page_num == 1:
        p1_primary = [(ch, 0) for ch in DIGITS + PUNCTUATION]
        p1_primary_rows = (len(p1_primary) + TEMPLATE_SPEC["cell_cols"] - 1) // TEMPLATE_SPEC["cell_cols"]
        sep_y_mm = (TEMPLATE_SPEC["margin_top_mm"]
                    + p1_primary_rows * TEMPLATE_SPEC["cell_height_mm"]
                    + 2)
        sep_y_pt = ph_pt - sep_y_mm * MM_TO_PT
        c.setStrokeColor(colors.HexColor("#999999"))
        c.setDash(3, 3)
        c.line(TEMPLATE_SPEC["margin_left_mm"] * MM_TO_PT, sep_y_pt,
               (pw_pt - TEMPLATE_SPEC["margin_left_mm"] * MM_TO_PT), sep_y_pt)
        c.setDash()
        c.setFont("Helvetica-Oblique", 7)
        c.setFillColor(colors.HexColor("#666666"))
        c.drawString(
            TEMPLATE_SPEC["margin_left_mm"] * MM_TO_PT,
            sep_y_pt - 4 * MM_TO_PT,
            "Alternates — write additional versions of the letters below for a more natural font"
        )

    # Draw cells
    for cell in cells_on_page:
        x_pt = cell.x_mm * MM_TO_PT
        # reportlab y=0 is bottom; convert from top-origin mm
        y_top_pt = ph_pt - (cell.y_mm + TEMPLATE_SPEC["cell_height_mm"]) * MM_TO_PT
        y_bottom_pt = ph_pt - cell.y_mm * MM_TO_PT

        # Outer cell border — very faint grid line
        c.setStrokeColor(colors.HexColor("#DDDDDD"))
        c.setLineWidth(0.3)
        c.rect(x_pt, y_top_pt, cw_pt, ch_pt, fill=0, stroke=1)

        # Inner writing box — full cell width, CAP line top, descender line bottom.
        # No marks of any kind inside the box. Extraction crops here.
        top_ratio     = TEMPLATE_SPEC["guideline_top_ratio"]
        bot_ratio     = TEMPLATE_SPEC["guideline_bottom_ratio"]
        box_top_pt    = y_top_pt + ch_pt * (1 - top_ratio)
        box_bottom_pt = y_top_pt + ch_pt * (1 - bot_ratio)
        box_h_pt      = box_top_pt - box_bottom_pt

        c.setStrokeColor(colors.HexColor("#AAAAAA"))
        c.setLineWidth(0.5)
        c.rect(x_pt, box_bottom_pt, cw_pt, box_h_pt, fill=0, stroke=1)

        # Character label — in the top gap above the inner box
        label = cell.char if cell.slot == 0 else f"{cell.char}·{cell.slot}"
        c.setFont("Helvetica", 5)
        c.setFillColor(colors.HexColor("#888888"))
        c.drawString(x_pt + 1 * MM_TO_PT, box_top_pt + 1 * MM_TO_PT, label)

    # x-height and baseline reference lines — drawn ONCE PER ROW in the far left
    # page margin (left of all cells). Never inside any cell or inner box.
    margin_right_pt = TEMPLATE_SPEC["margin_left_mm"] * MM_TO_PT
    label_x_pt = 2 * MM_TO_PT
    drawn_rows = set()
    for cell in cells_on_page:
        if cell.row in drawn_rows:
            continue
        drawn_rows.add(cell.row)
        cell_top_mm = TEMPLATE_SPEC["margin_top_mm"] + cell.row * TEMPLATE_SPEC["cell_height_mm"]
        cell_h_mm   = TEMPLATE_SPEC["cell_height_mm"]
        for ratio, ref_label in [
            (TEMPLATE_SPEC["guideline_xheight_ratio"], "x"),
            (TEMPLATE_SPEC["guideline_baseline_ratio"], "base"),
        ]:
            ref_y_pt = ph_pt - (cell_top_mm + cell_h_mm * ratio) * MM_TO_PT
            c.setStrokeColor(colors.HexColor("#BBBBBB"))
            c.setLineWidth(0.4)
            c.line(label_x_pt + 3 * MM_TO_PT, ref_y_pt, margin_right_pt, ref_y_pt)
            c.setFont("Helvetica", 3.5)
            c.setFillColor(colors.HexColor("#BBBBBB"))
            c.drawString(label_x_pt, ref_y_pt + 0.5, ref_label)

    # Footer
    c.setFont("Helvetica", 6)
    c.setFillColor(colors.HexColor("#888888"))
    c.drawString(
        TEMPLATE_SPEC["margin_left_mm"] * MM_TO_PT,
        5 * MM_TO_PT,
        "Keep all four corner markers fully visible in your scan or photo."
    )
    page_label = f"Page {page_num + 1} of 2"
    c.drawRightString(
        (TEMPLATE_SPEC["page_width_mm"] - TEMPLATE_SPEC["margin_left_mm"]) * MM_TO_PT,
        5 * MM_TO_PT,
        page_label
    )


def generate_template_pdf() -> bytes:
    """Generate the handwriting template PDF and return as bytes."""
    buf = io.BytesIO()
    pw = TEMPLATE_SPEC["page_width_mm"] * MM_TO_PT
    ph = TEMPLATE_SPEC["page_height_mm"] * MM_TO_PT
    c = rl_canvas.Canvas(buf, pagesize=(pw, ph))

    cells_p0 = [cell for cell in CELL_LAYOUT if cell.page == 0]
    cells_p1 = [cell for cell in CELL_LAYOUT if cell.page == 1]

    _draw_page(c, 0, cells_p0)
    c.showPage()
    _draw_page(c, 1, cells_p1)
    c.showPage()
    c.save()

    return buf.getvalue()


# Cache the template bytes so we only generate once
_template_cache: bytes | None = None


def get_template_pdf() -> bytes:
    global _template_cache
    if _template_cache is None:
        _template_cache = generate_template_pdf()
    return _template_cache
