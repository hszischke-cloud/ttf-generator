"""
alignment.py — Registration marker detection and perspective correction.

Takes a raw page image (numpy array) and returns a corrected image aligned
to the canonical TEMPLATE_SPEC coordinate space.
"""

import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

from template import TEMPLATE_SPEC, MM_TO_PX, DPI


class AlignmentError(Exception):
    """Raised when registration markers cannot be found."""
    pass


# Expected marker positions in mm from top-left of page (center of each marker)
def _expected_marker_positions_px() -> List[Tuple[float, float]]:
    """Return expected marker centers in pixels (top-left origin) at 300 dpi."""
    pad = TEMPLATE_SPEC["marker_padding_mm"]
    r = TEMPLATE_SPEC["marker_diameter_mm"] / 2
    center_offset = pad + r  # distance from page edge to marker center
    pw = TEMPLATE_SPEC["page_width_mm"]
    ph = TEMPLATE_SPEC["page_height_mm"]
    # Order: top-left, top-right, bottom-left, bottom-right
    return [
        (center_offset * MM_TO_PX,          center_offset * MM_TO_PX),
        ((pw - center_offset) * MM_TO_PX,   center_offset * MM_TO_PX),
        (center_offset * MM_TO_PX,          (ph - center_offset) * MM_TO_PX),
        ((pw - center_offset) * MM_TO_PX,   (ph - center_offset) * MM_TO_PX),
    ]


def load_image_from_file(path: str) -> np.ndarray:
    """Load an image file, applying EXIF rotation, return as BGR numpy array."""
    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img)
    pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def find_registration_markers(img_gray: np.ndarray) -> List[Tuple[float, float]]:
    """
    Detect the 4 registration markers in a grayscale image.

    Returns a list of 4 (x, y) centers sorted as:
      [top-left, top-right, bottom-left, bottom-right]

    Raises AlignmentError if exactly 4 markers cannot be found.
    """
    h, w = img_gray.shape

    # Threshold: markers are solid black on white background
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Expected marker size range at 300 dpi
    expected_r_px = (TEMPLATE_SPEC["marker_diameter_mm"] / 2) * MM_TO_PX
    min_area = (expected_r_px * 0.4) ** 2 * np.pi
    max_area = (expected_r_px * 1.8) ** 2 * np.pi

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.75:
            continue

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        candidates.append((cx, cy, circularity, area))

    if len(candidates) < 4:
        raise AlignmentError(
            f"Found only {len(candidates)} registration markers (need 4). "
            "Make sure all four corner markers are visible and unobstructed."
        )

    # Sort by circularity descending, take best 4
    candidates.sort(key=lambda c: -c[2])
    best4 = [(cx, cy) for cx, cy, _, _ in candidates[:4]]

    # Sort into [top-left, top-right, bottom-left, bottom-right]
    sorted_markers = _sort_markers(best4)
    return sorted_markers


def _sort_markers(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Sort 4 points into [top-left, top-right, bottom-left, bottom-right]."""
    pts = sorted(points, key=lambda p: p[1])  # sort by y
    top = sorted(pts[:2], key=lambda p: p[0])
    bottom = sorted(pts[2:], key=lambda p: p[0])
    return [top[0], top[1], bottom[0], bottom[1]]  # TL, TR, BL, BR


def correct_perspective(
    img: np.ndarray,
    markers: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Apply perspective correction using detected marker positions vs expected positions.

    Args:
        img: BGR image (any resolution)
        markers: [top-left, top-right, bottom-left, bottom-right] centers in pixels

    Returns:
        Corrected image at canonical 300 dpi page size.
    """
    dst_pts = np.array(_expected_marker_positions_px(), dtype=np.float32)
    src_pts = np.array(markers, dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    out_w = int(TEMPLATE_SPEC["page_width_mm"] * MM_TO_PX)
    out_h = int(TEMPLATE_SPEC["page_height_mm"] * MM_TO_PX)

    corrected = cv2.warpPerspective(img, H, (out_w, out_h),
                                    flags=cv2.INTER_LANCZOS4,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255))
    return corrected


def align_page(img_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    """
    Full alignment pipeline for one page image.

    Returns:
        (aligned_img, error_message)
        If error_message is not None, alignment failed and aligned_img is the
        original image (for manual fallback in the frontend).
    """
    # Resolution check
    h, w = img_bgr.shape[:2]
    if w < 1200 or h < 1600:
        return img_bgr, (
            f"Image resolution too low ({w}×{h}px). "
            "Please scan or photograph at higher resolution (minimum 1200×1600)."
        )

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    try:
        markers = find_registration_markers(gray)
    except AlignmentError as e:
        return img_bgr, str(e)

    aligned = correct_perspective(img_bgr, markers)
    return aligned, None


def align_page_with_manual_points(
    img_bgr: np.ndarray,
    manual_points: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Perspective correction using user-provided corner points.

    Args:
        img_bgr: original page image
        manual_points: [top-left, top-right, bottom-left, bottom-right]
                       pixel coordinates clicked by user in the frontend

    Returns:
        Corrected image at canonical 300 dpi page size.
    """
    return correct_perspective(img_bgr, manual_points)
