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
    fea_warning: Optional[str] = None
    line_skipped_glyphs: List[str] = []
    has_line_font: bool = False


class GlyphInfo(BaseModel):
    glyph_id: str          # e.g. "e_0", "e_1", "A_0"
    char: str
    slot: int              # 0 = primary, 1+ = alternate
    image_b64: str         # base64-encoded PNG of the drawn glyph
    accepted: bool = True


class GlyphsResponse(BaseModel):
    job_id: str
    glyphs: List[GlyphInfo]
    alignment_warnings: List[str] = []  # legacy; always empty in draw-only flow


class FinalizeRequest(BaseModel):
    approved_glyph_ids: List[str]
    font_name: str
    font_style: str = "Regular"
    letter_spacing: int = 0
    space_width: int = 600


class FinalizeResponse(BaseModel):
    job_id: str
    otf_url: str
    woff2_url: str
    otf_line_url: str
    woff2_line_url: str


class DrawGlyphRequest(BaseModel):
    """One drawn glyph from the canvas."""
    glyph_id: str           # e.g. "a_0", "a_init", "a_medi"
    char: str
    slot: int               # 0 = primary, 1+ = alternate (cycled by calt)
    svg_paths: List[str]    # filled brush outline paths "M x y L x y C ... Z" (dimensional font)
    pen_paths: List[List[List[float]]]  # raw pen tracks: list of strokes; each stroke is [[x,y], ...]
    svg_width: int          # canvas width in px
    svg_height: int         # canvas height in px
    baseline_y: int         # canvas y of the baseline guideline
    upscale_factor: float = 1.0
    thumbnail_png_b64: str  # base64 PNG for review

    # Cursive-mode metadata (None / 0 in print mode)
    form: str = "iso"       # one of: iso (isolated), init, medi, fina
    entry_y: Optional[float] = None  # y in canvas coords where this glyph wants its left connection
    exit_y: Optional[float] = None   # y where the right connection should land
