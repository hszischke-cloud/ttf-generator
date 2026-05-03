"""
centerline.py — Extract single-line centerlines from binary glyph images.

For pen-plotter / engraving use: each pen stroke = one polyline. Open
polylines emit as 'M L L L' (no Z) so the OTF→SVG round-trip preserves them
as open paths; closed shapes (o, 0, e/a/d bowls) emit with a trailing Z.

Pipeline:
  binary glyph (255=ink) → upscale (shared with vectorize.py)
  → skimage skeletonize → 8-connected pixel graph
  → walk between endpoints/junctions, then trace remaining cycles
  → drop spurs / micro-segments → Douglas-Peucker simplify
  → emit SVG paths (open or closed)
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.morphology import skeletonize

from processing.vectorize import _upscale_for_tracing

# Pruning thresholds, expressed as fractions of upscaled glyph height (px).
# Polylines are classified by endpoint type so each kind gets a fair threshold:
#   spur     — one degree-1 endpoint (could be noise sticking out of a junction)
#   isolated — both ends are degree-1 endpoints (complete simple stroke)
#   bridge   — both ends are degree-3+ junctions (mandatory link between strokes)
#   cycle    — a closed loop with no endpoints
SPUR_FRACTION = 0.015         # endpoint→junction min length (was 0.05; too aggressive)
ISOLATED_FRACTION = 0.015     # endpoint→endpoint min length
CYCLE_FRACTION = 0.025        # closed loop min length (was 0.03)
BRIDGE_MIN_PX = 5             # junction→junction absolute floor (kills 1-3px junction-cluster artefacts but keeps real bridges)

# Douglas-Peucker simplification (relative to glyph height in pixels)
DP_EPSILON_FACTOR = 0.005

# 8-connected neighbour offsets (dy, dx)
_NEIGHBOURS = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]


# ---------------------------------------------------------------------------
# Skeleton + graph helpers
# ---------------------------------------------------------------------------

def _build_skeleton(glyph_img: np.ndarray) -> np.ndarray:
    """Return a boolean array where True = skeleton pixel."""
    blurred = cv2.GaussianBlur(glyph_img, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return skeletonize(binary > 0)


def _neighbour_count(skel: np.ndarray, y: int, x: int) -> int:
    h, w = skel.shape
    n = 0
    for dy, dx in _NEIGHBOURS:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
            n += 1
    return n


def _walk(
    skel: np.ndarray,
    visited: np.ndarray,
    start: Tuple[int, int],
    first: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """
    Walk along the skeleton from `start` (an endpoint or junction) through
    `first` and onward. Stops at the next endpoint/junction or dead end.

    Marks only degree-2 (interior) pixels as visited so endpoints/junctions
    can serve as start points for multiple polylines.
    """
    h, w = skel.shape
    pts: List[Tuple[int, int]] = [start, first]

    deg_first = _neighbour_count(skel, first[0], first[1])
    if deg_first != 2:
        # Single-segment polyline — first is itself an endpoint or junction.
        return pts

    visited[first[0], first[1]] = True
    prev: Tuple[int, int] = start
    cur: Tuple[int, int] = first

    while True:
        next_p: Optional[Tuple[int, int]] = None
        for dy, dx in _NEIGHBOURS:
            ny, nx = cur[0] + dy, cur[1] + dx
            if (ny, nx) == prev:
                continue
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and not visited[ny, nx]:
                next_p = (ny, nx)
                break

        if next_p is None:
            return pts

        pts.append(next_p)
        deg = _neighbour_count(skel, next_p[0], next_p[1])
        if deg != 2:
            # Reached endpoint or junction — leave its visited bit alone.
            return pts

        visited[next_p[0], next_p[1]] = True
        prev = cur
        cur = next_p


def _extract_polylines(
    skel: np.ndarray,
) -> List[Tuple[List[Tuple[int, int]], bool]]:
    """
    Walk the skeleton graph and extract every polyline.

    Returns list of (points, is_closed). Closed polylines have the start
    point repeated at the end.
    """
    h, w = skel.shape
    visited = np.zeros_like(skel, dtype=bool)

    skel_pts = list(zip(*np.where(skel)))
    endpoints: List[Tuple[int, int]] = []
    junctions: List[Tuple[int, int]] = []
    for (y, x) in skel_pts:
        deg = _neighbour_count(skel, int(y), int(x))
        if deg == 1:
            endpoints.append((int(y), int(x)))
        elif deg >= 3:
            junctions.append((int(y), int(x)))

    polylines: List[Tuple[List[Tuple[int, int]], bool]] = []

    # 1. Walk from every endpoint
    for ep in endpoints:
        first: Optional[Tuple[int, int]] = None
        for dy, dx in _NEIGHBOURS:
            ny, nx = ep[0] + dy, ep[1] + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                first = (ny, nx)
                break
        if first is None:
            continue
        if visited[first[0], first[1]]:
            # The line was already walked from the other endpoint.
            continue
        pts = _walk(skel, visited, ep, first)
        polylines.append((pts, False))

    # 2. Walk every junction along each unvisited skeleton neighbour
    for jn in junctions:
        for dy, dx in _NEIGHBOURS:
            ny, nx = jn[0] + dy, jn[1] + dx
            if not (0 <= ny < h and 0 <= nx < w and skel[ny, nx]):
                continue
            if visited[ny, nx]:
                continue
            pts = _walk(skel, visited, jn, (ny, nx))
            if len(pts) >= 2:
                polylines.append((pts, False))

    # 3. Pure cycles (no endpoint, no junction) — pick a pixel, walk around
    for (y, x) in skel_pts:
        y, x = int(y), int(x)
        if visited[y, x]:
            continue
        if _neighbour_count(skel, y, x) != 2:
            continue
        first = None
        for dy, dx in _NEIGHBOURS:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx] and not visited[ny, nx]:
                first = (ny, nx)
                break
        if first is None:
            continue
        visited[y, x] = True
        pts = _walk(skel, visited, (y, x), first)
        if pts and pts[-1] != (y, x):
            pts.append((y, x))
        polylines.append((pts, True))

    # 4. Auto-detect closure: any polyline whose endpoints coincide is closed
    finalised: List[Tuple[List[Tuple[int, int]], bool]] = []
    for pts, closed in polylines:
        is_closed = closed or (len(pts) > 2 and pts[0] == pts[-1])
        finalised.append((pts, is_closed))
    return finalised


def _polyline_length(pts: List[Tuple[int, int]]) -> float:
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for (y1, x1), (y2, x2) in zip(pts, pts[1:]):
        total += float(np.hypot(y2 - y1, x2 - x1))
    return total


def _classify_polyline(
    pts: List[Tuple[int, int]],
    closed: bool,
    skel: np.ndarray,
) -> str:
    """
    Classify a polyline by the topology of its endpoints in the skeleton.

    'cycle'    — closed loop (no real endpoints in the graph)
    'isolated' — both ends are degree-1 endpoints (a complete simple stroke)
    'bridge'   — both ends are degree-3+ junctions (link between two strokes)
    'spur'     — exactly one end is a degree-1 endpoint, other is a junction
    'unknown'  — anything else (treated like spur for pruning)
    """
    if closed:
        return 'cycle'
    if len(pts) < 2:
        return 'unknown'
    deg_a = _neighbour_count(skel, pts[0][0], pts[0][1])
    deg_b = _neighbour_count(skel, pts[-1][0], pts[-1][1])
    a_end, b_end = deg_a == 1, deg_b == 1
    a_jct, b_jct = deg_a >= 3, deg_b >= 3
    if a_end and b_end:
        return 'isolated'
    if a_jct and b_jct:
        return 'bridge'
    if (a_end and b_jct) or (a_jct and b_end):
        return 'spur'
    return 'unknown'


def _douglas_peucker(
    pts: List[Tuple[int, int]],
    epsilon: float,
    closed: bool,
) -> List[Tuple[int, int]]:
    if len(pts) < 3:
        return pts
    arr = np.array([[p[1], p[0]] for p in pts], dtype=np.int32).reshape(-1, 1, 2)
    simplified = cv2.approxPolyDP(arr, epsilon, closed=closed)
    return [(int(p[0][1]), int(p[0][0])) for p in simplified]


def _polyline_to_svg(pts: List[Tuple[int, int]], closed: bool) -> str:
    """
    Emit an SVG path string suitable for OTF storage and pen-plotter export.

    CFF auto-closes every subpath, so there's no true "open contour" in OTF.
    For open polylines we retrace forward then backward — this gives a
    zero-area degenerate closed contour whose implicit closing segment has
    zero length. When the user converts the OTF to SVG, the pen plotter draws
    each stroke twice (acceptable) instead of drawing a phantom closing line
    from the polyline's end back to its start (broken).

    A 1-unit diagonal jog is inserted at the tip to break T2CharStringPen's
    hlineto/vlineto axis-aware merging — without it, the forward and back
    pure-horizontal/vertical legs collapse into zero-delta moves and points
    drop out of the charstring.

    Closed polylines (loops like 'o') emit normally with a trailing Z.
    """
    if len(pts) < 2:
        return ""
    parts = [f"M {pts[0][1]} {pts[0][0]}"]
    for y, x in pts[1:]:
        parts.append(f"L {x} {y}")
    if closed:
        return " ".join(parts) + " Z"
    tip_y, tip_x = pts[-1]
    parts.append(f"L {tip_x + 1} {tip_y + 1}")
    for y, x in reversed(pts[:-1]):
        parts.append(f"L {x} {y}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def vectorize_centerline(
    glyph_img: np.ndarray,
) -> Optional[Tuple[List[str], int, int, float]]:
    """
    Extract single-line centerlines from a binary glyph image.

    Returns (svg_paths, width_px, height_px, upscale_factor) — same shape as
    `vectorize_glyph()` so the font builder can consume either output
    interchangeably. None if no usable strokes were found.
    """
    if glyph_img is None or glyph_img.size == 0:
        return None
    if int(np.count_nonzero(glyph_img)) < 20:
        return None

    h_orig = glyph_img.shape[0]
    upscaled = _upscale_for_tracing(glyph_img)
    h, w = upscaled.shape
    upscale_factor = h / h_orig if h_orig > 0 else 1.0

    skel = _build_skeleton(upscaled)
    if not skel.any():
        return None

    polylines = _extract_polylines(skel)
    if not polylines:
        return None

    # Per-classification length thresholds (see constants above for rationale)
    thresholds = {
        'spur':     max(3.0, h * SPUR_FRACTION),
        'isolated': max(3.0, h * ISOLATED_FRACTION),
        'bridge':   float(BRIDGE_MIN_PX),
        'cycle':    max(3.0, h * CYCLE_FRACTION),
        'unknown':  max(3.0, h * SPUR_FRACTION),
    }

    pruned: List[Tuple[List[Tuple[int, int]], bool]] = []
    for pts, closed in polylines:
        length = _polyline_length(pts)
        kind = _classify_polyline(pts, closed, skel)
        if length < thresholds[kind]:
            continue
        pruned.append((pts, closed))

    if not pruned:
        return None

    epsilon = max(1.0, h * DP_EPSILON_FACTOR)
    simplified: List[Tuple[List[Tuple[int, int]], bool]] = []
    for pts, closed in pruned:
        s = _douglas_peucker(pts, epsilon, closed)
        if len(s) >= 2:
            simplified.append((s, closed))

    if not simplified:
        return None

    # Plotter-friendly stroke order: top-to-bottom, then left-to-right by start point
    simplified.sort(key=lambda pc: (pc[0][0][0], pc[0][0][1]))

    svg_paths = [_polyline_to_svg(pts, closed) for pts, closed in simplified]
    svg_paths = [p for p in svg_paths if p]
    if not svg_paths:
        return None

    return svg_paths, w, h, upscale_factor
