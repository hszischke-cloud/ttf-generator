"""Reorder SVG vectors for efficient pen-plotter travel.

A pen plotter draws one stroke at a time, lifting the pen and moving (a "travel"
or pen-up move) between strokes. If the strokes are drawn in the order they
happen to appear in the file, the pen can criss-cross the page wastefully.

This module takes an SVG, splits it into independent strokes (each ``M``-started
subpath, plus ``line``/``polyline``/``polygon``/``rect``/``circle``/``ellipse``
elements), then reorders them so the pen always heads to the *nearest* next
stroke. Open strokes may be reversed so the pen enters at whichever end is
closer. The drawn geometry is preserved exactly &mdash; only the order and the
direction of travel change.

The heuristic is greedy nearest-neighbour (the "closest vector next" rule),
followed by an optional bounded 2-opt pass that untangles obvious crossings.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

Point = Tuple[float, float]

SVG_NS = "http://www.w3.org/2000/svg"

# Style attributes worth carrying from the source onto each emitted path so the
# reordered SVG still looks like the original.
_STYLE_KEYS = (
    "stroke",
    "stroke-width",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-dasharray",
    "stroke-opacity",
    "opacity",
    "fill",
)

_TOKEN_RE = re.compile(
    r"([MmLlHhVvCcSsQqTtAaZz])|([-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?)"
)
_FLOAT_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")


@dataclass
class Stroke:
    """One continuous pen-down run: drawn from ``start`` to ``end``."""

    start: Point
    end: Point
    closed: bool
    segments: list  # normalized absolute segments (see _parse_subpaths)
    style: Dict[str, str] = field(default_factory=dict)


@dataclass
class OptimizeResult:
    svg: str
    width: float
    height: float
    path_count: int
    original_travel: float
    optimized_travel: float
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Path "d" parsing
# --------------------------------------------------------------------------- #
def _tokenize_path(d: str):
    tokens = []
    for m in _TOKEN_RE.finditer(d):
        if m.group(1):
            tokens.append(m.group(1))
        else:
            tokens.append(float(m.group(2)))
    return tokens


def _parse_subpaths(d: str) -> List[dict]:
    """Parse an SVG path ``d`` string into a list of absolute-coordinate subpaths.

    Each subpath is ``{"start", "end", "closed", "segments"}`` where every
    segment is one of:

    * ``("L", (x, y))``
    * ``("C", (x1, y1, x2, y2, x, y))``
    * ``("Q", (x1, y1, x, y))``
    * ``("A", (rx, ry, rot, large, sweep, x, y))``

    All shorthand commands (H/V/S/T and relative variants) are expanded to these
    absolute forms so strokes can be reversed and re-emitted uniformly.
    """
    tokens = _tokenize_path(d)
    n = len(tokens)
    subpaths: List[dict] = []
    state = {"i": 0}
    cur: Optional[dict] = None
    cx = cy = sx = sy = 0.0
    last_base = None
    last_ctrl: Optional[Point] = None
    cmd: Optional[str] = None

    def rd() -> float:
        v = tokens[state["i"]]
        state["i"] += 1
        return float(v)

    while state["i"] < n:
        tok = tokens[state["i"]]
        if isinstance(tok, str):
            cmd = tok
            state["i"] += 1
        if cmd is None:
            state["i"] += 1
            continue

        base = cmd.upper()
        rel = cmd.islower()

        if base == "Z":
            if cur is not None:
                cur["closed"] = True
                cx, cy = sx, sy
                cur["end"] = (cx, cy)
                subpaths.append(cur)
                cur = None
            last_base = "Z"
            cmd = None  # a stray coord after Z is ignored until the next command
            continue

        if base == "M":
            if cur is not None and cur["segments"]:
                cur["end"] = (cx, cy)
                subpaths.append(cur)
            x = rd()
            y = rd()
            if rel:
                x += cx
                y += cy
            cx, cy = x, y
            sx, sy = x, y
            cur = {"start": (x, y), "segments": [], "closed": False, "end": (x, y)}
            cmd = "l" if rel else "L"  # implicit repeats after M are lineto
            last_base = "M"
            last_ctrl = None
            continue

        if cur is None:  # a drawing command before any moveto
            cur = {"start": (cx, cy), "segments": [], "closed": False, "end": (cx, cy)}
            sx, sy = cx, cy

        if base == "L":
            x = rd()
            y = rd()
            if rel:
                x += cx
                y += cy
            cur["segments"].append(("L", (x, y)))
            cx, cy = x, y
            last_ctrl = None
        elif base == "H":
            x = rd()
            if rel:
                x += cx
            cur["segments"].append(("L", (x, cy)))
            cx = x
            last_ctrl = None
        elif base == "V":
            y = rd()
            if rel:
                y += cy
            cur["segments"].append(("L", (cx, y)))
            cy = y
            last_ctrl = None
        elif base == "C":
            x1, y1, x2, y2, x, y = rd(), rd(), rd(), rd(), rd(), rd()
            if rel:
                x1 += cx; y1 += cy; x2 += cx; y2 += cy; x += cx; y += cy
            cur["segments"].append(("C", (x1, y1, x2, y2, x, y)))
            last_ctrl = (x2, y2)
            cx, cy = x, y
        elif base == "S":
            x2, y2, x, y = rd(), rd(), rd(), rd()
            if rel:
                x2 += cx; y2 += cy; x += cx; y += cy
            if last_base in ("C", "S") and last_ctrl is not None:
                x1, y1 = 2 * cx - last_ctrl[0], 2 * cy - last_ctrl[1]
            else:
                x1, y1 = cx, cy
            cur["segments"].append(("C", (x1, y1, x2, y2, x, y)))
            last_ctrl = (x2, y2)
            cx, cy = x, y
        elif base == "Q":
            x1, y1, x, y = rd(), rd(), rd(), rd()
            if rel:
                x1 += cx; y1 += cy; x += cx; y += cy
            cur["segments"].append(("Q", (x1, y1, x, y)))
            last_ctrl = (x1, y1)
            cx, cy = x, y
        elif base == "T":
            x, y = rd(), rd()
            if rel:
                x += cx; y += cy
            if last_base in ("Q", "T") and last_ctrl is not None:
                x1, y1 = 2 * cx - last_ctrl[0], 2 * cy - last_ctrl[1]
            else:
                x1, y1 = cx, cy
            cur["segments"].append(("Q", (x1, y1, x, y)))
            last_ctrl = (x1, y1)
            cx, cy = x, y
        elif base == "A":
            rx, ry, rot, large, sweep, x, y = (
                rd(), rd(), rd(), rd(), rd(), rd(), rd()
            )
            if rel:
                x += cx; y += cy
            cur["segments"].append(("A", (rx, ry, rot, large, sweep, x, y)))
            cx, cy = x, y
            last_ctrl = None
        else:  # unknown token; advance to avoid an infinite loop
            state["i"] += 1

        last_base = base

    if cur is not None and cur["segments"]:
        cur["end"] = (cx, cy)
        subpaths.append(cur)

    return subpaths


def _reverse_segments(start: Point, segments: list) -> Tuple[Point, list]:
    """Return ``(new_start, segments)`` drawing the subpath end-to-start."""
    # Absolute endpoint reached after each segment, starting from the move point.
    points = [start]
    for seg in segments:
        kind, vals = seg
        points.append((vals[-2], vals[-1]))

    new_start = points[-1]
    out = []
    for k in range(len(segments) - 1, -1, -1):
        kind, vals = segments[k]
        from_pt = points[k]  # where this segment originally began
        if kind == "L":
            out.append(("L", (from_pt[0], from_pt[1])))
        elif kind == "C":
            x1, y1, x2, y2, _, _ = vals
            out.append(("C", (x2, y2, x1, y1, from_pt[0], from_pt[1])))
        elif kind == "Q":
            x1, y1, _, _ = vals
            out.append(("Q", (x1, y1, from_pt[0], from_pt[1])))
        elif kind == "A":
            rx, ry, rot, large, sweep, _, _ = vals
            # Reversing an arc flips its sweep direction.
            out.append(
                ("A", (rx, ry, rot, large, 1 - int(sweep), from_pt[0], from_pt[1]))
            )
    return new_start, out


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #
def _n(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _segments_to_d(start: Point, segments: list, closed: bool) -> str:
    parts = [f"M {_n(start[0])} {_n(start[1])}"]
    for kind, vals in segments:
        if kind == "L":
            parts.append(f"L {_n(vals[0])} {_n(vals[1])}")
        elif kind == "C":
            parts.append("C " + " ".join(_n(v) for v in vals))
        elif kind == "Q":
            parts.append("Q " + " ".join(_n(v) for v in vals))
        elif kind == "A":
            rx, ry, rot, large, sweep, x, y = vals
            parts.append(
                f"A {_n(rx)} {_n(ry)} {_n(rot)} "
                f"{int(large)} {int(sweep)} {_n(x)} {_n(y)}"
            )
    if closed:
        parts.append("Z")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# SVG element extraction
# --------------------------------------------------------------------------- #
def _localname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _floats(s: str) -> List[float]:
    return [float(x) for x in _FLOAT_RE.findall(s or "")]


def _element_style(el: ET.Element) -> Dict[str, str]:
    style: Dict[str, str] = {}
    inline = el.get("style")
    if inline:
        for decl in inline.split(";"):
            if ":" in decl:
                k, v = decl.split(":", 1)
                k = k.strip()
                if k in _STYLE_KEYS:
                    style[k] = v.strip()
    for key in _STYLE_KEYS:
        val = el.get(key)
        if val is not None:
            style[key] = val
    return style


def _points_to_subpath(pts: List[Point], closed: bool) -> Optional[dict]:
    if len(pts) < 2:
        return None
    segments = [("L", (x, y)) for x, y in pts[1:]]
    return {
        "start": pts[0],
        "end": pts[-1],
        "closed": closed,
        "segments": segments,
    }


def _shape_subpaths(el: ET.Element, tag: str) -> List[dict]:
    """Convert a basic shape element into absolute-coordinate subpaths."""
    if tag == "path":
        return _parse_subpaths(el.get("d", ""))

    if tag == "line":
        x1 = float(el.get("x1", 0)); y1 = float(el.get("y1", 0))
        x2 = float(el.get("x2", 0)); y2 = float(el.get("y2", 0))
        sp = _points_to_subpath([(x1, y1), (x2, y2)], closed=False)
        return [sp] if sp else []

    if tag in ("polyline", "polygon"):
        nums = _floats(el.get("points", ""))
        pts = list(zip(nums[0::2], nums[1::2]))
        sp = _points_to_subpath(pts, closed=(tag == "polygon"))
        return [sp] if sp else []

    if tag == "rect":
        x = float(el.get("x", 0)); y = float(el.get("y", 0))
        w = float(el.get("width", 0)); h = float(el.get("height", 0))
        if w <= 0 or h <= 0:
            return []
        pts = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        sp = _points_to_subpath(pts, closed=True)
        return [sp] if sp else []

    if tag in ("circle", "ellipse"):
        cx = float(el.get("cx", 0)); cy = float(el.get("cy", 0))
        if tag == "circle":
            rx = ry = float(el.get("r", 0))
        else:
            rx = float(el.get("rx", 0)); ry = float(el.get("ry", 0))
        if rx <= 0 or ry <= 0:
            return []
        # Two half-arcs make a full closed ellipse.
        start = (cx - rx, cy)
        segments = [
            ("A", (rx, ry, 0, 0, 1, cx + rx, cy)),
            ("A", (rx, ry, 0, 0, 1, cx - rx, cy)),
        ]
        return [{"start": start, "end": start, "closed": True, "segments": segments}]

    return []


def _collect_strokes(root: ET.Element) -> List[Stroke]:
    strokes: List[Stroke] = []

    def walk(el: ET.Element, inherited: Dict[str, str]):
        style = dict(inherited)
        style.update(_element_style(el))
        tag = _localname(el.tag)
        if tag in ("path", "line", "polyline", "polygon", "rect", "circle", "ellipse"):
            for sp in _shape_subpaths(el, tag):
                if not sp["segments"]:
                    continue
                strokes.append(
                    Stroke(
                        start=sp["start"],
                        end=sp["end"],
                        closed=sp.get("closed", False),
                        segments=sp["segments"],
                        style=dict(style),
                    )
                )
        for child in list(el):
            walk(child, style)

    walk(root, {})
    return strokes


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #
def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _travel(strokes: List[Stroke], order: List[Tuple[int, bool]], start: Point) -> float:
    """Total pen-up distance for a given draw order/direction."""
    pen = start
    total = 0.0
    for idx, rev in order:
        s = strokes[idx]
        entry = s.end if rev else s.start
        exit_pt = s.start if rev else s.end
        total += _dist(pen, entry)
        pen = exit_pt
    return total


def _nearest_neighbour(
    strokes: List[Stroke], indices: List[int], start: Point, allow_reverse: bool
) -> Tuple[List[Tuple[int, bool]], Point]:
    """Greedy "closest vector next" ordering. Returns (order, final pen pos)."""
    remaining = set(indices)
    order: List[Tuple[int, bool]] = []
    pen = start
    while remaining:
        best_i = -1
        best_d = math.inf
        best_rev = False
        for i in remaining:
            s = strokes[i]
            d_start = _dist(pen, s.start)
            if d_start < best_d:
                best_d, best_i, best_rev = d_start, i, False
            if allow_reverse and not s.closed:
                d_end = _dist(pen, s.end)
                if d_end < best_d:
                    best_d, best_i, best_rev = d_end, i, True
        remaining.discard(best_i)
        order.append((best_i, best_rev))
        s = strokes[best_i]
        pen = s.start if best_rev else s.end
    return order, pen


def _two_opt(
    strokes: List[Stroke],
    order: List[Tuple[int, bool]],
    start: Point,
    allow_reverse: bool,
    max_passes: int = 6,
) -> List[Tuple[int, bool]]:
    """Bounded 2-opt: reverse contiguous blocks (flipping their direction) when
    that shortens total travel. Skipped for large inputs to stay responsive."""
    n = len(order)
    if n < 4 or n > 400:
        return order

    def flip(seg):
        idx, rev = seg
        return (idx, (not rev) if allow_reverse and not strokes[idx].closed else rev)

    best = _travel(strokes, order, start)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                candidate = (
                    order[:i]
                    + [flip(s) for s in reversed(order[i:j + 1])]
                    + order[j + 1:]
                )
                cand_travel = _travel(strokes, candidate, start)
                if cand_travel + 1e-9 < best:
                    order = candidate
                    best = cand_travel
                    improved = True
    return order


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def _svg_dimensions(root: ET.Element) -> Tuple[float, float, Optional[str]]:
    viewbox = root.get("viewBox")
    w = root.get("width")
    h = root.get("height")
    width = _floats(w)[0] if w and _floats(w) else None
    height = _floats(h)[0] if h and _floats(h) else None
    if (width is None or height is None) and viewbox:
        vb = _floats(viewbox)
        if len(vb) == 4:
            width = width if width is not None else vb[2]
            height = height if height is not None else vb[3]
    return (width or 0.0, height or 0.0, viewbox)


def _build_svg(
    strokes: List[Stroke],
    order: List[Tuple[int, bool]],
    width: float,
    height: float,
    viewbox: Optional[str],
) -> str:
    vb = viewbox or f"0 0 {_n(width)} {_n(height)}"
    head = (
        f'<svg xmlns="{SVG_NS}" '
        f'width="{_n(width)}" height="{_n(height)}" viewBox="{vb}">'
    )
    lines = [head]
    for idx, rev in order:
        s = strokes[idx]
        start, segments = s.start, s.segments
        if rev and not s.closed:
            start, segments = _reverse_segments(s.start, s.segments)
        d = _segments_to_d(start, segments, s.closed)
        attrs = {
            "fill": s.style.get("fill", "none"),
            "stroke": s.style.get("stroke", "#000000"),
            "stroke-width": s.style.get("stroke-width", "1"),
        }
        for key in ("stroke-linecap", "stroke-linejoin", "stroke-dasharray", "stroke-opacity", "opacity"):
            if key in s.style:
                attrs[key] = s.style[key]
        attr_str = " ".join(f'{k}="{v}"' for k, v in attrs.items())
        lines.append(f'<path d="{d}" {attr_str}/>')
    lines.append("</svg>")
    return "\n".join(lines)


def optimize_svg(
    svg_text: str,
    *,
    allow_reverse: bool = True,
    group_by_color: bool = True,
    two_opt: bool = True,
    start: Point = (0.0, 0.0),
) -> OptimizeResult:
    """Reorder the strokes of an SVG to minimise pen-plotter travel.

    Parameters
    ----------
    allow_reverse:
        Let an open stroke be drawn from either end (enter at the closer one).
    group_by_color:
        Keep same-coloured strokes together and optimise within each colour
        before moving on &mdash; minimises pen swaps on a multi-pen plotter.
    two_opt:
        Run a bounded 2-opt refinement after nearest-neighbour ordering.
    """
    try:
        root = ET.fromstring(svg_text)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse SVG: {exc}") from exc

    strokes = _collect_strokes(root)
    width, height, viewbox = _svg_dimensions(root)

    if not strokes:
        return OptimizeResult(
            svg=svg_text,
            width=width,
            height=height,
            path_count=0,
            original_travel=0.0,
            optimized_travel=0.0,
            stats={"strokes": 0},
        )

    original_order = [(i, False) for i in range(len(strokes))]
    original_travel = _travel(strokes, original_order, start)

    if group_by_color:
        groups: Dict[str, List[int]] = {}
        group_order: List[str] = []
        for i, s in enumerate(strokes):
            color = s.style.get("stroke", "#000000")
            if color not in groups:
                groups[color] = []
                group_order.append(color)
            groups[color].append(i)
        order: List[Tuple[int, bool]] = []
        pen = start
        for color in group_order:
            entry = pen  # pen position when this colour group begins
            grp_order, pen = _nearest_neighbour(strokes, groups[color], entry, allow_reverse)
            if two_opt:
                grp_order = _two_opt(strokes, grp_order, entry, allow_reverse)
                pen = _final_pen(strokes, grp_order, entry)
            order.extend(grp_order)
    else:
        order, _ = _nearest_neighbour(
            strokes, list(range(len(strokes))), start, allow_reverse
        )
        if two_opt:
            order = _two_opt(strokes, order, start, allow_reverse)

    optimized_travel = _travel(strokes, order, start)

    svg = _build_svg(strokes, order, width, height, viewbox)

    improvement = (
        100.0 * (original_travel - optimized_travel) / original_travel
        if original_travel > 0
        else 0.0
    )

    return OptimizeResult(
        svg=svg,
        width=width,
        height=height,
        path_count=len(strokes),
        original_travel=original_travel,
        optimized_travel=optimized_travel,
        stats={
            "strokes": len(strokes),
            "closed_strokes": sum(1 for s in strokes if s.closed),
            "colors": len({s.style.get("stroke", "#000000") for s in strokes}),
            "improvement_pct": round(improvement, 1),
        },
    )


def _final_pen(strokes: List[Stroke], order: List[Tuple[int, bool]], start: Point) -> Point:
    pen = start
    for idx, rev in order:
        s = strokes[idx]
        pen = s.start if (rev and not s.closed) else s.end
    return pen
