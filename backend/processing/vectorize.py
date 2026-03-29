"""
vectorize.py — Convert binary glyph images to SVG path data.

Uses OpenCV contour detection + Douglas-Peucker simplification to produce
SVG path strings. Pure Python/NumPy — no external binaries, no segfaults.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Minimum glyph height in pixels before tracing (upscale if smaller)
MIN_TRACE_HEIGHT_PX = 400


def _upscale_for_tracing(glyph_img: np.ndarray) -> np.ndarray:
    h, w = glyph_img.shape[:2]
    if h < MIN_TRACE_HEIGHT_PX:
        scale = MIN_TRACE_HEIGHT_PX / h
        glyph_img = cv2.resize(
            glyph_img,
            (int(w * scale), MIN_TRACE_HEIGHT_PX),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return glyph_img


def _contour_to_svg_path(contour: np.ndarray, epsilon_factor: float = 0.005) -> str:
    """
    Convert an OpenCV contour to an SVG path 'd' string using
    Douglas-Peucker simplification for smooth-ish curves.
    """
    epsilon = epsilon_factor * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) < 3:
        return ""

    pts = approx.reshape(-1, 2)
    d = f"M {pts[0][0]} {pts[0][1]}"
    for pt in pts[1:]:
        d += f" L {pt[0]} {pt[1]}"
    d += " Z"
    return d


def vectorize_glyph(
    glyph_img: np.ndarray,
) -> Optional[Tuple[List[str], int, int, float]]:
    """
    Convert a binary glyph image to SVG path data using OpenCV contours.

    Args:
        glyph_img: 2D numpy array, binary (255=ink, 0=background)

    Returns:
        (svg_path_list, width_px, height_px, upscale_factor) or None if no contours found.
        upscale_factor: ratio of upscaled height to original height (1.0 if no upscale).
    """
    if glyph_img is None or glyph_img.size == 0:
        return None

    # Guard: skip near-blank images
    ink_pixels = int(np.count_nonzero(glyph_img))
    if ink_pixels < 20:
        return None

    h_orig = glyph_img.shape[0]
    upscaled = _upscale_for_tracing(glyph_img)
    h, w = upscaled.shape
    upscale_factor = h / h_orig if h_orig > 0 else 1.0

    # Slight blur then re-threshold to smooth jagged pixel edges
    blurred = cv2.GaussianBlur(upscaled, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
    )

    if not contours:
        return None

    # Filter tiny noise contours (< 0.1% of image area)
    min_area = h * w * 0.001
    svg_paths = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        path = _contour_to_svg_path(cnt)
        if path:
            svg_paths.append(path)

    if not svg_paths:
        return None

    return svg_paths, w, h, upscale_factor
