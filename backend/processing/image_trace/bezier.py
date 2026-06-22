"""Polyline simplification and smooth curve generation.

Two steps turn a noisy pixel polyline into a compact, smooth SVG path:

1. ``rdp`` (Ramer-Douglas-Peucker) drops points that lie close to the line
   between their neighbours, dramatically reducing the vertex count.
2. ``catmull_rom_to_bezier`` converts the simplified polyline into a chain of
   cubic Bezier segments that pass through every remaining point, giving smooth
   curves instead of visible polygon corners.
"""

from __future__ import annotations

from typing import List, Tuple

Point = Tuple[float, float]


def rdp(points: List[Point], epsilon: float) -> List[Point]:
    """Ramer-Douglas-Peucker simplification (iterative to avoid recursion limits)."""
    if epsilon <= 0 or len(points) < 3:
        return list(points)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        start, end = stack.pop()
        ax, ay = points[start]
        bx, by = points[end]
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy

        dmax, index = 0.0, start
        for i in range(start + 1, end):
            px, py = points[i]
            if seg_len_sq == 0:
                dist = ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
            else:
                # Perpendicular distance from point to segment AB.
                cross = abs(dx * (ay - py) - dy * (ax - px))
                dist = cross / seg_len_sq ** 0.5
            if dist > dmax:
                dmax, index = dist, i

        if dmax > epsilon:
            keep[index] = True
            stack.append((start, index))
            stack.append((index, end))

    return [p for p, k in zip(points, keep) if k]


def catmull_rom_to_bezier(points: List[Point], tension: float = 1.0) -> str:
    """Build an SVG path ``d`` string of cubic Beziers through ``points``.

    ``tension`` in ``[0, 1]`` scales how much the curve bows: 0 yields straight
    line segments, 1 yields a standard (smooth) Catmull-Rom spline.
    """
    if not points:
        return ""
    if len(points) == 1:
        x, y = points[0]
        return f"M {_n(x)} {_n(y)}"
    if len(points) == 2:
        (x0, y0), (x1, y1) = points
        return f"M {_n(x0)} {_n(y0)} L {_n(x1)} {_n(y1)}"

    closed = points[0] == points[-1]
    pts = points[:-1] if closed else points
    n = len(pts)

    def get(i):
        if closed:
            return pts[i % n]
        return pts[min(max(i, 0), n - 1)]

    x0, y0 = pts[0]
    d = [f"M {_n(x0)} {_n(y0)}"]

    last = n if closed else n - 1
    for i in range(last):
        p0 = get(i - 1)
        p1 = get(i)
        p2 = get(i + 1)
        p3 = get(i + 2)

        # Catmull-Rom -> cubic Bezier control points.
        c1x = p1[0] + (p2[0] - p0[0]) / 6.0 * tension
        c1y = p1[1] + (p2[1] - p0[1]) / 6.0 * tension
        c2x = p2[0] - (p3[0] - p1[0]) / 6.0 * tension
        c2y = p2[1] - (p3[1] - p1[1]) / 6.0 * tension

        d.append(
            f"C {_n(c1x)} {_n(c1y)} {_n(c2x)} {_n(c2y)} {_n(p2[0])} {_n(p2[1])}"
        )

    if closed:
        d.append("Z")
    return " ".join(d)


def _n(value: float) -> str:
    """Format a number compactly (trim trailing zeros, max 2 decimals)."""
    return f"{value:.2f}".rstrip("0").rstrip(".")
