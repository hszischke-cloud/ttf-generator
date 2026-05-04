"""
models.py — Pydantic request/response models for the FastAPI API.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    AWAITING_REVIEW = "awaiting_review"
    FINALIZING = "finalizing"
    COMPLETE = "complete"
    ERROR = "error"


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress_pct: int = 0
    error_message: Optional[str] = None
    # Set when alignment fails — base64 page image for manual corner selection
    alignment_failed_image_b64: Optional[str] = None
    # Set if feaLib failed to compile OpenType features (calt/ss01/ss02 will be missing)
    fea_warning: Optional[str] = None
    # Glyph IDs that couldn't be centerline-traced (skipped in the line font)
    line_skipped_glyphs: List[str] = []
    # True only if the single-line companion font was actually built and saved
    has_line_font: bool = False


class GlyphInfo(BaseModel):
    glyph_id: str          # e.g. "e_0", "e_1", "A_0"
    char: str              # the character this glyph represents
    slot: int              # 0 = primary, 1+ = alternate
    image_b64: str         # base64-encoded PNG of the tight-cropped glyph
    accepted: bool = True  # default: accept all


class GlyphsResponse(BaseModel):
    job_id: str
    glyphs: List[GlyphInfo]
    alignment_warnings: List[str] = []


class ManualAlignmentPoint(BaseModel):
    x: float
    y: float


class FinalizeRequest(BaseModel):
    approved_glyph_ids: List[str]
    font_name: str
    font_style: str = "Regular"
    letter_spacing: int = 0    # extra UPM units added to every glyph's advance width
    space_width: int = 600     # advance width for the space glyph in UPM units
    # Optional manual alignment points per page (if auto-detection failed)
    manual_alignment: Optional[List[List[ManualAlignmentPoint]]] = None


class FinalizeResponse(BaseModel):
    job_id: str
    otf_url: str
    woff2_url: str
    # Single-line / centerline companion font (for pen plotters)
    otf_line_url: str
    woff2_line_url: str


class DrawGlyphRequest(BaseModel):
    glyph_id: str           # e.g. "a_0", "a_1"
    char: str
    slot: int               # 0 = primary, 1+ = alternate
    svg_paths: List[str]    # filled outline paths "M x y L x y C ... Z"
    svg_width: int          # canvas width in px
    svg_height: int         # canvas height in px
    baseline_y: int         # canvas.height * 0.72 (baseline guideline Y)
    upscale_factor: float = 1.0
    thumbnail_png_b64: str  # base64 PNG for review page
