"""
models.py — Pydantic request/response models for the FastAPI API.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


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
    # Public CDN URL of the thumbnail PNG. The browser loads thumbnails
    # directly (parallel + cacheable) instead of the backend proxying every
    # PNG through base64 — that proxying made the review page scale linearly
    # with glyph count.
    image_url: Optional[str] = None
    image_b64: Optional[str] = None  # legacy fallback; no longer populated
    accepted: bool = True


class GlyphsResponse(BaseModel):
    job_id: str
    glyphs: List[GlyphInfo]
    alignment_warnings: List[str] = []  # legacy; always empty in draw-only flow


class GlyphBearing(BaseModel):
    """Per-glyph manual side-bearing override (UPM units).

    lsb = left side bearing (gap from the glyph origin to the leftmost ink).
    rsb = right side bearing (gap from the rightmost ink to the advance).
    Only meaningful for iso (print) glyphs; cursive forms derive their
    bearings from connection points and ignore these.
    """
    lsb: int
    rsb: int


class FinalizeRequest(BaseModel):
    approved_glyph_ids: List[str]
    font_name: str
    font_style: str = "Regular"
    letter_spacing: int = 0
    space_width: int = 600
    # Per-glyph side-bearing overrides keyed by glyph_id. When None, existing
    # overrides in job state are preserved (so a plain spacing re-finalize
    # doesn't wipe out border adjustments).
    glyph_bearings: Optional[Dict[str, GlyphBearing]] = None


class FinalizeResponse(BaseModel):
    job_id: str
    otf_url: str
    otf_line_url: str


class SavedFontInfo(BaseModel):
    job_id: str
    font_name: str
    mode: str        # "print" or "cursive"
    glyph_count: int
    saved_at: float  # Unix timestamp


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
    # x positions in submitted (post-xShift) canvas coords where the user
    # wants the connection to land. Backend uses these in bearing math
    # instead of assuming canvas PAD, which lets each letter dial in its
    # own connection x to fit its natural width.
    entry_x: Optional[float] = None
    exit_x: Optional[float] = None
    entry_y: Optional[float] = None
    exit_y: Optional[float] = None
    x_shift: float = 0.0    # minX - PAD applied when building pen_paths/svg_paths

    # Font-level style settings — same for every glyph in a font
    pen_tool: str = "pen"
    pen_size: int = 14
    pen_color: List[int] = Field(default_factory=lambda: [38, 32, 28])


class DrawGlyphBatchRequest(BaseModel):
    """Many drawn glyphs in one request — replaces N serial /glyph POSTs.

    Submitting glyphs one-by-one was O(n²): every request re-read and re-wrote
    the whole (multi-MB) manifest just to append one entry. A batch is one
    manifest read + one write, with thumbnail uploads fanned out in parallel.
    """
    glyphs: List[DrawGlyphRequest]
