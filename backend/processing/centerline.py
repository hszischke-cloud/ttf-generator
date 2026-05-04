"""
centerline.py — Convert raw pen-path polylines into single-line SVG paths
suitable for OTF storage and pen-plotter SVG export.

Phase 0 implementation: emit each stroke as `M L L L` with a 1-unit
perpendicular tip jog and forward-back retrace, so OTF→SVG conversion in
the user's downstream tool gives a usable single-line path. Phase 1 will
replace the straight segments with smooth Bezier curves fit through the
pen path.

Skeletonization-based extraction (used in the photo flow) is gone — we now
capture the user's actual pen path from the canvas, which is a perfect
lossless centerline.
"""

from typing import List, Optional, Sequence


def _polyline_to_svg(pts: Sequence[Sequence[float]]) -> str:
    """
    Emit one stroke as an SVG path string. Open polylines are forward-back
    retraced with a 1-unit perpendicular tip jog so:

      - the implicit close (CFF auto-closes every subpath) is a no-op;
      - T2CharStringPen's hlineto/vlineto axis-merging doesn't cancel
        opposing same-axis lineTos and silently drop points.

    `pts` is a list of [x, y] (canvas coordinates). Returns "" if too short.
    """
    if len(pts) < 2:
        return ""
    parts = [f"M {pts[0][0]} {pts[0][1]}"]
    for x, y in pts[1:]:
        parts.append(f"L {x} {y}")
    tip_x, tip_y = pts[-1]
    parts.append(f"L {tip_x + 1} {tip_y + 1}")
    for x, y in reversed(pts[:-1]):
        parts.append(f"L {x} {y}")
    return " ".join(parts)


def polyline_paths_to_svg(strokes: Sequence[Sequence[Sequence[float]]]) -> List[str]:
    """
    Convert a list of pen-stroke polylines (one polyline per stroke, each
    polyline a list of [x, y] points) to a list of SVG path strings.

    Strokes shorter than 2 points or empty are dropped silently.
    """
    out: List[str] = []
    for stroke in strokes:
        if not stroke:
            continue
        d = _polyline_to_svg(stroke)
        if d:
            out.append(d)
    return out
