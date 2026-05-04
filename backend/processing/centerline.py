"""
centerline.py — Convert raw pen-path polylines into smooth single-line SVG
paths suitable for OTF storage and pen-plotter SVG export.

The user's mouse/pen track from the canvas IS the centerline — perfect,
lossless, no skeletonization needed. We do three things:

  1. Dedupe nearly-coincident points (canvas captures redundant samples).
  2. Detect sharp corners (direction change > threshold) so they stay as
     polyline vertices instead of getting rounded by curve fitting.
  3. Fit centripetal Catmull-Rom Beziers through the smooth runs between
     corners, converted to cubic Beziers (`M C C C ...`).

The emitted SVG uses the same forward-back retrace + 1-unit tip jog trick
as before so OTF/CFF round-trips don't break the open-contour intent: the
pen plotter sees one continuous single-line path per stroke.
"""

import math
from typing import List, Sequence, Tuple

# Drop consecutive samples closer than this (in canvas px). Real handwriting
# detail is way coarser than the canvas sample rate.
MIN_POINT_DISTANCE = 2.0

# Douglas-Peucker simplification tolerance (canvas px). Removes points that
# fall within this distance of the line between their neighbours — kills the
# noise from a 60-Hz canvas sampling a slow pen drag.
DP_EPSILON = 1.5

# Direction change above this threshold is treated as a corner — preserved
# as a polyline vertex so curve fitting doesn't soften it.
CORNER_ANGLE_DEG = 60.0

# Catmull-Rom tension. 0.5 gives the standard centripetal CR.
TENSION = 0.5

Point = Tuple[float, float]


# ---------------------------------------------------------------------------
# Smoothing pipeline
# ---------------------------------------------------------------------------

def _dedupe(pts: Sequence[Sequence[float]], min_dist: float = MIN_POINT_DISTANCE) -> List[Point]:
    if not pts:
        return []
    out: List[Point] = [(float(pts[0][0]), float(pts[0][1]))]
    md2 = min_dist * min_dist
    for p in pts[1:]:
        last = out[-1]
        dx, dy = float(p[0]) - last[0], float(p[1]) - last[1]
        if dx * dx + dy * dy >= md2:
            out.append((float(p[0]), float(p[1])))
    return out


def _perp_distance(p: Point, a: Point, b: Point) -> float:
    """Perpendicular distance from p to the line through a and b."""
    abx, aby = b[0] - a[0], b[1] - a[1]
    n = math.hypot(abx, aby)
    if n == 0:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    cross = (abx) * (a[1] - p[1]) - (a[0] - p[0]) * (aby)
    return abs(cross) / n


def _douglas_peucker(pts: List[Point], epsilon: float) -> List[Point]:
    """Iterative Ramer-Douglas-Peucker simplification (avoids recursion limit)."""
    if len(pts) < 3:
        return list(pts)
    keep = [False] * len(pts)
    keep[0] = True
    keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        max_d, max_i = 0.0, lo + 1
        a, b = pts[lo], pts[hi]
        for i in range(lo + 1, hi):
            d = _perp_distance(pts[i], a, b)
            if d > max_d:
                max_d, max_i = d, i
        if max_d > epsilon:
            keep[max_i] = True
            stack.append((lo, max_i))
            stack.append((max_i, hi))
    return [p for p, k in zip(pts, keep) if k]


def _angle_change_deg(a: Point, b: Point, c: Point) -> float:
    v1x, v1y = b[0] - a[0], b[1] - a[1]
    v2x, v2y = c[0] - b[0], c[1] - b[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 == 0 or n2 == 0:
        return 0.0
    dot = (v1x * v2x + v1y * v2y) / (n1 * n2)
    if dot > 1.0: dot = 1.0
    if dot < -1.0: dot = -1.0
    return math.degrees(math.acos(dot))


def _split_at_corners(pts: List[Point]) -> List[List[Point]]:
    """Split the polyline into smooth runs at sharp corners."""
    if len(pts) <= 2:
        return [pts] if pts else []
    runs: List[List[Point]] = []
    cur: List[Point] = [pts[0]]
    for i in range(1, len(pts) - 1):
        cur.append(pts[i])
        if _angle_change_deg(pts[i - 1], pts[i], pts[i + 1]) > CORNER_ANGLE_DEG:
            runs.append(cur)
            cur = [pts[i]]
    cur.append(pts[-1])
    runs.append(cur)
    return runs


def _catmull_rom_to_bezier(
    p0: Point, p1: Point, p2: Point, p3: Point, tension: float = TENSION
) -> Tuple[Point, Point]:
    """Centripetal Catmull-Rom: control points for the cubic Bezier from p1 to p2."""
    c1 = (p1[0] + (p2[0] - p0[0]) * tension / 3.0,
          p1[1] + (p2[1] - p0[1]) * tension / 3.0)
    c2 = (p2[0] - (p3[0] - p1[0]) * tension / 3.0,
          p2[1] - (p3[1] - p1[1]) * tension / 3.0)
    return c1, c2


# Segment representations:
#   ('line', target_point)
#   ('cubic', control1, control2, target_point)
Segment = Tuple

def _fit_run(run: List[Point]) -> List[Segment]:
    """Fit a smooth run (between corners or at ends) to Bezier or line segments."""
    if len(run) < 2:
        return []
    if len(run) == 2:
        return [('line', run[1])]

    # Phantom endpoints by mirror reflection — gives natural-looking ends
    phantom_start = (2 * run[0][0] - run[1][0], 2 * run[0][1] - run[1][1])
    phantom_end = (2 * run[-1][0] - run[-2][0], 2 * run[-1][1] - run[-2][1])
    extended = [phantom_start] + run + [phantom_end]

    segs: List[Segment] = []
    for i in range(1, len(extended) - 2):
        p0, p1, p2, p3 = extended[i - 1], extended[i], extended[i + 1], extended[i + 2]
        c1, c2 = _catmull_rom_to_bezier(p0, p1, p2, p3)
        segs.append(('cubic', c1, c2, p2))
    return segs


def _smooth_polyline_to_segments(pts: Sequence[Sequence[float]]) -> Tuple[Point, List[Segment]]:
    """
    Convert a raw pen path to (start_point, [segments]).

    Returns ((0,0), []) for empty / too-short input — caller should treat
    that as no contribution to the line font.
    """
    cleaned = _dedupe(pts)
    if len(cleaned) < 2:
        return ((0.0, 0.0), [])
    cleaned = _douglas_peucker(cleaned, DP_EPSILON)
    runs = _split_at_corners(cleaned)
    segs: List[Segment] = []
    for run in runs:
        segs.extend(_fit_run(run))
    return (cleaned[0], segs)


# ---------------------------------------------------------------------------
# SVG emission with retrace + tip jog
# ---------------------------------------------------------------------------

def _seg_target(seg: Segment) -> Point:
    return seg[-1]  # last element is always the target point


def _reverse_segments(start: Point, segs: List[Segment]) -> List[Segment]:
    """
    Reverse a sequence of segments: traverse the same curve from end to start.
    A cubic (c1, c2, P_end) starting at P_prev becomes (c2, c1, P_prev) when
    traversed in reverse. A line just keeps its target swapped.
    """
    if not segs:
        return []
    # Build the list of waypoints: [start, end1, end2, ..., endN]
    pts: List[Point] = [start]
    for seg in segs:
        pts.append(_seg_target(seg))
    out: List[Segment] = []
    # For each forward segment i (from pts[i] to pts[i+1]), the reversed
    # segment goes from pts[i+1] to pts[i].
    for i in range(len(segs) - 1, -1, -1):
        seg = segs[i]
        target = pts[i]
        if seg[0] == 'line':
            out.append(('line', target))
        elif seg[0] == 'cubic':
            _, c1, c2, _ = seg
            out.append(('cubic', c2, c1, target))
    return out


def _segments_to_svg_parts(segs: List[Segment]) -> List[str]:
    parts: List[str] = []
    for seg in segs:
        if seg[0] == 'line':
            _, p = seg
            parts.append(f"L {p[0]:.2f} {p[1]:.2f}")
        elif seg[0] == 'cubic':
            _, c1, c2, p = seg
            parts.append(
                f"C {c1[0]:.2f} {c1[1]:.2f} {c2[0]:.2f} {c2[1]:.2f} {p[0]:.2f} {p[1]:.2f}"
            )
    return parts


def _stroke_to_svg(pts: Sequence[Sequence[float]]) -> str:
    """One pen stroke → one SVG path string (forward + tip jog + reversed retrace)."""
    start, segs = _smooth_polyline_to_segments(pts)
    if not segs:
        return ""

    parts: List[str] = [f"M {start[0]:.2f} {start[1]:.2f}"]
    parts.extend(_segments_to_svg_parts(segs))

    # Tip jog (1px diagonal) — defeats T2CharStringPen axis-merging
    end = _seg_target(segs[-1])
    parts.append(f"L {end[0] + 1:.2f} {end[1] + 1:.2f}")

    # Reverse retrace through the same control points
    parts.extend(_segments_to_svg_parts(_reverse_segments(start, segs)))

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _stroke_endpoints(s: Sequence[Sequence[float]]) -> Tuple[Point, Point]:
    return (float(s[0][0]), float(s[0][1])), (float(s[-1][0]), float(s[-1][1]))


def _dist2(a: Point, b: Point) -> float:
    dx, dy = a[0] - b[0], a[1] - b[1]
    return dx * dx + dy * dy


def _optimise_stroke_order(
    strokes: Sequence[Sequence[Sequence[float]]],
) -> List[List[List[float]]]:
    """
    Reorder strokes (and reverse individual ones if it helps) so the pen
    plotter minimises pen-up travel between them.

    Greedy nearest-neighbour starting from the stroke endpoint closest to
    top-left. Centripetal Catmull-Rom is symmetric under point reversal
    so reversing a stroke before curve fitting traces the same curve in
    the opposite direction — pen travel changes, visual result doesn't.
    """
    pool = [list(s) for s in strokes if s]
    if len(pool) <= 1:
        return pool

    # Pick the starting stroke: whichever has its closest-to-top-left
    # endpoint. We orient that stroke so it begins at top-left.
    origin = (0.0, 0.0)
    best_i, best_rev, best_d = 0, False, float('inf')
    for i, s in enumerate(pool):
        a, b = _stroke_endpoints(s)
        da = _dist2(a, origin)
        db = _dist2(b, origin)
        if da < best_d:
            best_d, best_i, best_rev = da, i, False
        if db < best_d:
            best_d, best_i, best_rev = db, i, True

    first = pool.pop(best_i)
    if best_rev:
        first.reverse()
    out: List[List[List[float]]] = [first]
    pen = (float(first[-1][0]), float(first[-1][1]))

    while pool:
        best_i, best_rev, best_d = 0, False, float('inf')
        for i, s in enumerate(pool):
            a, b = _stroke_endpoints(s)
            da = _dist2(a, pen)
            db = _dist2(b, pen)
            if da < best_d:
                best_d, best_i, best_rev = da, i, False
            if db < best_d:
                best_d, best_i, best_rev = db, i, True
        s = pool.pop(best_i)
        if best_rev:
            s.reverse()
        out.append(s)
        pen = (float(s[-1][0]), float(s[-1][1]))
    return out


def polyline_paths_to_svg(strokes: Sequence[Sequence[Sequence[float]]]) -> List[str]:
    """
    Convert a list of pen-stroke polylines (one polyline per stroke, each
    a list of [x, y] points) into smooth-Bezier SVG path strings.

    Strokes are reordered via greedy nearest-neighbour TSP first so the
    pen plotter doesn't waste time travelling between far-apart strokes.

    Each emitted path uses the forward + tip-jog + reverse-retrace pattern
    so it round-trips cleanly through CFF and exports as a usable
    single-line path.
    """
    ordered = _optimise_stroke_order(strokes)
    out: List[str] = []
    for stroke in ordered:
        if not stroke:
            continue
        d = _stroke_to_svg(stroke)
        if d:
            out.append(d)
    return out
