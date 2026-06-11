"""
autospace.py — optical side-bearing suggestions (perceived-width equalization).

Implements an HT-Letterspacer-style margin analysis over each glyph's outline
paths. For every iso (print) glyph we scan horizontal slices across a vertical
measurement zone and measure how much white space sits between the glyph's
bounding edge and its actual ink at each height. Averaging those margins gives
an "openness" score per side; the suggested bearing is then whatever makes the
total optical side space hit a common target:

    suggested = (target − average_margin) × px→UPM scale

So a bare stem (i, l: average margin ≈ 0) gets a positive bearing — extra
border space that props up its perceived width — while a letter with a big
overhanging arm (T, L, V: large white wedges under the arm) gets a negative
bearing that tucks the neighbouring letter in under the overhang.

The target is self-calibrating: the median margin across all letter sides in
the font, so a "typical" letter lands at ≈ 0 and everything else adjusts
relative to it. This keeps overall text colour the same as the edge-default
spacing the user already dialled in with the global letter-spacing slider.

These are *suggestions only* — the caller (border editor UI) presents them in
the editable lsb/rsb fields and nothing is persisted until the user applies.
"""

import re
import statistics
from typing import Dict, List, Optional, Tuple

from processing.font_builder import CELL_SCALE

# Guideline ratios — must match the drawing canvas (y = ratio × canvas height).
CAP_RATIO = 0.15
XHEIGHT_RATIO = 0.42
BASELINE_RATIO = 0.72

# Canvas padding around the ink bbox (same constant as font_builder.CANVAS_PAD;
# duplicated here so this module stays import-light for unit tests).
CANVAS_PAD = 12

# Horizontal scanlines sampled per glyph measurement zone.
SAMPLES = 64

# Cubic curve flattening steps (legacy vtracer paths; drawn paths are M/L/Z).
_CURVE_STEPS = 8

# Margins deeper than this fraction of the x-height band stop counting —
# without a cap, a deep counter (the open right side of an L) would drag the
# bearing arbitrarily negative.
_DEPTH_CAP_FRAC = 0.35

# Suggested bearings are clamped to this UPM range.
CLAMP_MIN = -150
CLAMP_MAX = 250

_TOKEN_RE = re.compile(
    r'[MLCZmlcz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
)


def _parse_polygons(d: str) -> List[List[Tuple[float, float]]]:
    """Parse an absolute M/L/C/Z SVG path into flattened point loops."""
    tokens = _TOKEN_RE.findall(d)
    polys: List[List[Tuple[float, float]]] = []
    pts: List[Tuple[float, float]] = []
    cmd = None
    i = 0
    n = len(tokens)

    def flush():
        nonlocal pts
        if len(pts) >= 3:
            polys.append(pts)
        pts = []

    while i < n:
        tok = tokens[i]
        if tok in 'MLCZmlcz':
            cmd = tok.upper()
            if cmd == 'Z':
                flush()
            i += 1
            continue
        try:
            if cmd == 'M' or cmd == 'L':
                x, y = float(tokens[i]), float(tokens[i + 1])
                if cmd == 'M' and pts:
                    flush()
                pts.append((x, y))
                i += 2
                # Repeated coordinate pairs after M behave like L.
                cmd = 'L' if cmd == 'M' else cmd
            elif cmd == 'C':
                if not pts:
                    i += 6
                    continue
                x0, y0 = pts[-1]
                c1x, c1y = float(tokens[i]), float(tokens[i + 1])
                c2x, c2y = float(tokens[i + 2]), float(tokens[i + 3])
                x1, y1 = float(tokens[i + 4]), float(tokens[i + 5])
                for s in range(1, _CURVE_STEPS + 1):
                    t = s / _CURVE_STEPS
                    mt = 1 - t
                    px = (mt ** 3 * x0 + 3 * mt ** 2 * t * c1x
                          + 3 * mt * t ** 2 * c2x + t ** 3 * x1)
                    py = (mt ** 3 * y0 + 3 * mt ** 2 * t * c1y
                          + 3 * mt * t ** 2 * c2y + t ** 3 * y1)
                    pts.append((px, py))
                i += 6
            else:
                i += 1
        except (ValueError, IndexError):
            break
    flush()
    return polys


def _row_extremes(
    polys: List[List[Tuple[float, float]]], y: float,
) -> Tuple[Optional[float], Optional[float]]:
    """Leftmost / rightmost outline crossing of the horizontal line at *y*."""
    lo = hi = None
    for poly in polys:
        m = len(poly)
        for i in range(m):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % m]
            if (y1 <= y < y2) or (y2 <= y < y1):
                t = (y - y1) / (y2 - y1)
                x = x1 + t * (x2 - x1)
                if lo is None or x < lo:
                    lo = x
                if hi is None or x > hi:
                    hi = x
    return lo, hi


def measure_glyph_margins(entry: Dict) -> Optional[Dict]:
    """
    Measure average left/right optical margins (canvas px) for one manifest
    entry. Returns {"avg_l", "avg_r", "coord_scale"} or None if the glyph has
    no usable outline.
    """
    polys: List[List[Tuple[float, float]]] = []
    for d in entry.get("svg_paths") or []:
        polys.extend(_parse_polygons(d))
    if not polys:
        return None

    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    min_y, max_y = min(ys), max(ys)
    if max_y - min_y < 2:
        return None

    H = entry.get("svg_height") or 400
    baseline = entry.get("baseline_y") or BASELINE_RATIO * H
    cap_y = CAP_RATIO * H
    xh_y = XHEIGHT_RATIO * H

    ch = entry.get("char") or ""
    if len(ch) == 1 and 'a' <= ch <= 'z':
        zone = (xh_y, baseline)
    elif len(ch) == 1 and ('A' <= ch <= 'Z' or ch.isdigit()):
        zone = (cap_y, baseline)
    else:
        # Punctuation/symbols: measure across the glyph's own ink band.
        zone = (min_y, max_y)

    # Clamp the zone to where ink actually exists; if the glyph doesn't reach
    # the zone at all (e.g. an apostrophe vs the x-height band) fall back to
    # its own ink band.
    z0, z1 = max(zone[0], min_y), min(zone[1], max_y)
    if z1 - z0 < 6:
        z0, z1 = min_y, max_y

    # Bearings are measured from the assumed ink box [PAD, svg_width − PAD]
    # (the same reference the font builder and border editor use).
    svg_w = entry.get("svg_width") or 0
    ref_l = float(CANVAS_PAD)
    ref_r = float(svg_w - CANVAS_PAD) if svg_w > 0 else max(xs)

    d_max = _DEPTH_CAP_FRAC * (BASELINE_RATIO - XHEIGHT_RATIO) * H

    left: List[float] = []
    right: List[float] = []
    for i in range(SAMPLES):
        y = z0 + (z1 - z0) * (i + 0.5) / SAMPLES
        lo, hi = _row_extremes(polys, y)
        if lo is None:
            # Fully open row (gap in the ink) counts as maximally open.
            left.append(d_max)
            right.append(d_max)
        else:
            left.append(min(max(lo - ref_l, 0.0), d_max))
            right.append(min(max(ref_r - hi, 0.0), d_max))

    upf = entry.get("upscale_factor") or 1.0
    cs = CELL_SCALE / upf if upf > 0 else CELL_SCALE
    return {
        "avg_l": sum(left) / len(left),
        "avg_r": sum(right) / len(right),
        "coord_scale": cs,
    }


def _clamp(v: int) -> int:
    return max(CLAMP_MIN, min(CLAMP_MAX, v))


def compute_auto_bearings(
    manifest: List[Dict], bias_upm: int = 0,
) -> Dict[str, Dict[str, int]]:
    """
    Suggest per-glyph (lsb, rsb) in UPM for every adjustable iso glyph in a
    job manifest. *bias_upm* shifts every suggestion looser (+) or tighter (−).
    """
    measured: List[Tuple[Dict, Dict]] = []
    for entry in manifest:
        if not entry.get("has_glyph"):
            continue
        if (entry.get("form") or "iso") != "iso":
            continue
        m = measure_glyph_margins(entry)
        if m is not None:
            measured.append((entry, m))

    if not measured:
        return {}

    # Self-calibrating target: median margin across letter sides, so a typical
    # letter stays at ≈ 0 adjustment and overall colour is preserved.
    letter_margins = [
        v
        for entry, m in measured
        if len(entry.get("char") or "") == 1 and entry["char"].isalpha()
        for v in (m["avg_l"], m["avg_r"])
    ]
    if len(letter_margins) >= 4:
        target = statistics.median(letter_margins)
    else:
        target = 8.0  # px — sane default for tiny glyph sets

    out: Dict[str, Dict[str, int]] = {}
    for entry, m in measured:
        cs = m["coord_scale"]
        lsb = _clamp(int(round((target - m["avg_l"]) * cs)) + bias_upm)
        rsb = _clamp(int(round((target - m["avg_r"]) * cs)) + bias_upm)
        out[entry["glyph_id"]] = {"lsb": lsb, "rsb": rsb}
    return out
