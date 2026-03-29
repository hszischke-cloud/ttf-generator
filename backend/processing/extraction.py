"""
extraction.py — Cell extraction and glyph segmentation.

Works on perspective-corrected page images aligned to TEMPLATE_SPEC canonical space.
Cell positions are computed arithmetically from TEMPLATE_SPEC (no image analysis needed
after alignment).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from template import TEMPLATE_SPEC, MM_TO_PX, CELL_LAYOUT, CellDef


@dataclass
class ExtractedGlyph:
    char: str
    slot: int                       # 0 = primary, 1+ = alternate
    glyph_id: str                   # e.g. "e_0", "e_1", "A_0"
    cell_img: np.ndarray            # full cell crop (grayscale)
    glyph_img: Optional[np.ndarray] # tight-cropped glyph binary (white on black), or None
    baseline_y_in_glyph: int        # y-coordinate of baseline within glyph_img


def _cell_rect_px(cell: CellDef) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) in pixels for a cell in canonical page space."""
    x = int(cell.x_mm * MM_TO_PX)
    y = int(cell.y_mm * MM_TO_PX)
    w = int(TEMPLATE_SPEC["cell_width_mm"] * MM_TO_PX)
    h = int(TEMPLATE_SPEC["cell_height_mm"] * MM_TO_PX)
    return x, y, w, h


def _inner_box_rect_px(cell: CellDef) -> Tuple[int, int, int, int]:
    """Return (x, y, w, h) of the inner writing box in pixels.

    The inner box is inset from the cell left by guide_gutter_mm and spans
    vertically from the CAP line to the descender line. Extraction crops to
    this region so gutter marks (x-height, baseline) and cell borders are
    never included.
    """
    cell_h_px  = int(TEMPLATE_SPEC["cell_height_mm"] * MM_TO_PX)
    cell_w_px  = int(TEMPLATE_SPEC["cell_width_mm"] * MM_TO_PX)
    top_offset = int(cell_h_px * TEMPLATE_SPEC["guideline_top_ratio"])
    bot_offset = int(cell_h_px * TEMPLATE_SPEC["guideline_bottom_ratio"])
    x = int(cell.x_mm * MM_TO_PX)
    y = int(cell.y_mm * MM_TO_PX) + top_offset
    return x, y, cell_w_px, bot_offset - top_offset


def _segment_glyph(
    cell_gray: np.ndarray,
    baseline_y_px: int,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Segment the handwritten glyph from a cell image.

    Returns:
        (glyph_img, baseline_y_in_glyph)
        glyph_img: tight-cropped binary image (255=ink, 0=background), or None if empty
        baseline_y_in_glyph: y-coordinate of baseline within glyph_img bounding box
    """
    baseline_y_in_cell = baseline_y_px

    # Gaussian blur to reduce noise before thresholding
    blurred = cv2.GaussianBlur(cell_gray, (3, 3), 0)

    # Adaptive threshold handles uneven illumination (phone photos, shadows)
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21,
        C=10
    )

    # Morphological closing to connect broken pencil/thin pen strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cell_area = cell_gray.shape[0] * cell_gray.shape[1]
    min_area = 50
    max_area = cell_area * 0.80

    valid_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        valid_boxes.append((x, y, x + w, y + h))

    if not valid_boxes:
        return None, baseline_y_in_cell

    # Merge all valid bounding boxes into one tight bounding box
    x1 = min(b[0] for b in valid_boxes)
    y1 = min(b[1] for b in valid_boxes)
    x2 = max(b[2] for b in valid_boxes)
    y2 = max(b[3] for b in valid_boxes)

    # Add padding (clamped to cell bounds)
    pad = 8
    cell_h, cell_w = cell_gray.shape
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(cell_w, x2 + pad)
    y2 = min(cell_h, y2 + pad)

    glyph_crop = binary[y1:y2, x1:x2]

    # Compute baseline position within the cropped glyph image
    baseline_y_in_glyph = max(0, baseline_y_in_cell - y1)

    return glyph_crop, baseline_y_in_glyph


def extract_glyphs_from_page(
    page_img_bgr: np.ndarray,
    page_num: int,
) -> List[ExtractedGlyph]:
    """
    Extract all glyphs from a single corrected page image.

    Args:
        page_img_bgr: BGR image aligned to canonical TEMPLATE_SPEC space
        page_num: 0 or 1

    Returns:
        List of ExtractedGlyph (one per cell on this page, including None-glyph for empty cells)
    """
    gray = cv2.cvtColor(page_img_bgr, cv2.COLOR_BGR2GRAY)
    page_cells = [cell for cell in CELL_LAYOUT if cell.page == page_num]
    results: List[ExtractedGlyph] = []

    # Pre-compute baseline y within the inner box (in pixels)
    cell_h_px  = int(TEMPLATE_SPEC["cell_height_mm"] * MM_TO_PX)
    top_offset = int(cell_h_px * TEMPLATE_SPEC["guideline_top_ratio"])
    baseline_y_px = int(cell_h_px * TEMPLATE_SPEC["guideline_baseline_ratio"]) - top_offset

    for cell in page_cells:
        x, y, w, h = _inner_box_rect_px(cell)

        # Bounds check (in case of small alignment errors)
        img_h, img_w = gray.shape
        x = max(0, min(x, img_w - 1))
        y = max(0, min(y, img_h - 1))
        x2 = max(0, min(x + w, img_w))
        y2 = max(0, min(y + h, img_h))

        cell_crop = gray[y:y2, x:x2]

        if cell_crop.size == 0:
            continue

        # Small uniform inset to exclude the inner box border line itself
        inset = 8
        cell_h_c, cell_w_c = cell_crop.shape
        if cell_h_c > inset * 2 and cell_w_c > inset * 2:
            cell_crop = cell_crop[inset:cell_h_c - inset, inset:cell_w_c - inset]

        glyph_img, baseline_y = _segment_glyph(cell_crop, baseline_y_px)

        glyph_id = f"{_safe_char_id(cell.char)}_{cell.slot}"
        results.append(ExtractedGlyph(
            char=cell.char,
            slot=cell.slot,
            glyph_id=glyph_id,
            cell_img=cell_crop,
            glyph_img=glyph_img,
            baseline_y_in_glyph=baseline_y,
        ))

    return results


def _safe_char_id(char: str) -> str:
    """Convert a character to a safe ASCII identifier for glyph IDs."""
    special = {
        '.': 'period', ',': 'comma', ';': 'semicolon', ':': 'colon',
        '!': 'exclam', '?': 'question', '"': 'quotedbl', "'": 'quotesingle',
        '(': 'parenleft', ')': 'parenright', '-': 'hyphen', '/': 'slash',
        '@': 'at', '&': 'ampersand', '#': 'numbersign', '%': 'percent',
    }
    return special.get(char, char)


def extract_all_glyphs(
    page_images: List[np.ndarray],
) -> List[ExtractedGlyph]:
    """
    Extract glyphs from all pages.

    Args:
        page_images: list of BGR page images (index = page number)

    Returns:
        Flat list of all extracted glyphs across all pages
    """
    all_glyphs: List[ExtractedGlyph] = []
    for page_num, page_img in enumerate(page_images):
        glyphs = extract_glyphs_from_page(page_img, page_num)
        all_glyphs.extend(glyphs)
    return all_glyphs
