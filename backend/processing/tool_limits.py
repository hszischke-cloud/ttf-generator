"""Centralized upload limits for the stateless pen-plotter / Cricut tools.

Every value is env-overridable so a deployment can tune them without a code
change. Keep these in sync with the helper text + client-side pre-checks in
app.html — the browser fetches them at runtime via GET /tools/limits, so the
UI and the server share this one source of truth.
"""

import os

_MB = 1024 * 1024


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


# --- Raster image uploads (POST /tools/image-to-svg) -----------------------
# Byte cap stops huge files at the door; the megapixel cap is the real DoS/OOM
# guard — a small compressed PNG can decode to hundreds of megapixels. Inputs
# are downscaled to MAX_TRACE_DIMENSION before tracing regardless.
MAX_IMAGE_BYTES = _env_int("TOOL_MAX_IMAGE_BYTES", 15 * _MB)
MAX_IMAGE_MEGAPIXELS = _env_float("TOOL_MAX_IMAGE_MEGAPIXELS", 40.0)
MAX_TRACE_DIMENSION = _env_int("TOOL_MAX_TRACE_DIMENSION", 2000)

# --- SVG uploads (POST /tools/optimize-svg) --------------------------------
MAX_SVG_BYTES = _env_int("TOOL_MAX_SVG_BYTES", 5 * _MB)

# PIL format names the tracer accepts, plus a friendly label for messages/UI.
ALLOWED_IMAGE_FORMATS = ("PNG", "JPEG", "GIF", "BMP", "WEBP", "TIFF")
ALLOWED_IMAGE_LABEL = "PNG, JPG, GIF, BMP, WEBP, TIFF"

# Hard ceiling for PIL's decompression-bomb guard. Set a little above the
# megapixel cap so a permitted image loads, but a malicious one still trips it.
PIL_MAX_IMAGE_PIXELS = int(MAX_IMAGE_MEGAPIXELS * 1_000_000 * 1.1)


def public_limits() -> dict:
    """The subset of limits exposed to the browser (GET /tools/limits)."""
    return {
        "max_image_bytes": MAX_IMAGE_BYTES,
        "max_image_mb": round(MAX_IMAGE_BYTES / _MB, 1),
        "max_image_megapixels": MAX_IMAGE_MEGAPIXELS,
        "max_svg_bytes": MAX_SVG_BYTES,
        "max_svg_mb": round(MAX_SVG_BYTES / _MB, 1),
        "allowed_image_formats": list(ALLOWED_IMAGE_FORMATS),
        "allowed_image_label": ALLOWED_IMAGE_LABEL,
        "max_trace_dimension": MAX_TRACE_DIMENSION,
    }
