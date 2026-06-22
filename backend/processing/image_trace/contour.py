"""Outline (filled-shape) tracing — the cut-file counterpart to centerline tracing.

Where ``skeleton.py`` traces the *centre* of each stroke (one line down the
middle, for pens / plotters / scoring), this traces the *outer edge* of the
inked regions as closed shapes (for cutting machines: Cricut, Silhouette,
vinyl). Both branches share the same preprocessing (threshold + despeckle ->
boolean mask) and the same downstream smoothing (RDP + Catmull-Rom Beziers),
so the two outputs stay visually consistent.
"""

from __future__ import annotations

from typing import List

import numpy as np
from skimage.measure import find_contours

from .bezier import catmull_rom_to_bezier, rdp


def _polyline_length(pts: List[tuple]) -> float:
    total = 0.0
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        total += ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
    return total


def mask_to_outline_paths(
    mask: np.ndarray,
    simplify: float = 1.5,
    smoothing: float = 1.0,
    min_path_length: int = 5,
) -> List[str]:
    """Trace the boundaries of the inked ``mask`` into closed Bezier ``d`` strings.

    Returns one ``d`` string per contour (outer boundaries *and* holes). The
    caller combines them into a single even-odd-filled path so holes subtract.
    """
    if not mask.any():
        return []

    # Pad by 1px so shapes that touch the image border still close into a ring
    # instead of being cut off by the edge. The offset is removed below.
    padded = np.pad(mask.astype(float), 1, mode="constant", constant_values=0.0)
    contours = find_contours(padded, 0.5)

    paths: List[str] = []
    for contour in contours:
        # contour points are (row, col) in padded space. Undo the pad and swap
        # to SVG (x=col, y=row).
        pts = [(float(c) - 1.0, float(r) - 1.0) for r, c in contour]
        if len(pts) < 3:
            continue
        # Make the ring explicitly closed so curve fitting emits a Z.
        if pts[0] != pts[-1]:
            pts.append(pts[0])
        if _polyline_length(pts) < max(0.0, float(min_path_length)):
            continue

        simplified = rdp(pts, simplify)
        if len(simplified) < 3:
            continue
        if simplified[0] != simplified[-1]:
            simplified.append(simplified[0])

        d = catmull_rom_to_bezier(simplified, smoothing)
        if d:
            paths.append(d)
    return paths
