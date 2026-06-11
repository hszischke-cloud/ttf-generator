"""
pen_realistic.py — "Realistic ink" stroker: raw pen-path centerlines →
outline SVG paths that read as real ballpoint/gel ink on paper.

A CFF OTF can only carry filled vector outlines (plus calt/ss alternates and
flat COLR colour) — no texture, alpha or gradients — so every property of
real pen writing has to be encoded as outline GEOMETRY. The model, in order
of visual importance:

1. Ink dynamics    — line width follows pen speed and pressure. A slow,
                     heavy passage runs wide and wet; a fast flick runs thin.
2. Ink pooling     — ink puddles where the pen dwells: stroke starts
                     (pen-down blob), slow lifts, and sharp corners (the
                     pen decelerates into and out of a turn).
3. Terminal shapes — a round tip leaves round caps; a fast lift leaves a
                     thin tapering flick tail; a slow lift leaves a small
                     pool before the pen leaves the paper.
4. Liquid edges    — outlines are smooth cubic Béziers (real ink never has
                     polygon facets), carrying subtle correlated waviness
                     plus occasional paper-fibre notches where the sheet's
                     tooth resists the ink.

Input strokes are the raw canvas pen tracks: ``[[x, y], ...]`` (legacy) or
``[[x, y, p, t], ...]`` with pressure ``p`` in 0..1 and stroke-relative
timestamps ``t`` in ms. When timestamps are missing, the spacing between raw
pointer samples doubles as a speed proxy (pointer events arrive at a roughly
fixed cadence, so distance-per-sample tracks velocity).

All randomness is value noise seeded from the glyph id, so rebuilding a font
reproduces identical outlines (content-versioned uploads stay cache-stable).

Output paths live in the same canvas-pixel coordinate space as the classic
``svg_paths``, so the rest of the font pipeline (baseline transform, advance
math, bearings) is untouched.
"""

import hashlib
import math
from typing import List, Optional, Sequence, Tuple

from processing.centerline import (
    _douglas_peucker,
    _fit_run,
    _perp_distance,
    _segments_to_svg_parts,
    _split_at_corners,
)

Point = Tuple[float, float]

# ── Ink model constants ─────────────────────────────────────────────────────
# Resampling step as a fraction of pen size (clamped in px below). Uniform
# arc-length sampling makes the noise, smoothing and pooling kernels stable
# regardless of how fast the pointer events arrived.
RESAMPLE_STEP_FRAC = 0.20
RESAMPLE_STEP_MIN_PX = 0.9
RESAMPLE_STEP_MAX_PX = 2.0

# Width = pen_size/2 × pressure_term × speed_term (× pooling/terminal envelopes)
PRESSURE_BASE = 0.78          # width multiplier at pressure 0
PRESSURE_GAIN = 0.42          # added at pressure 1 (p≈0.52 → ×1.0)
SPEED_EXPONENT = 0.22         # width ∝ (v/v_median)^-exp
SPEED_FACTOR_MIN = 0.78       # fast-stroke thinning floor
SPEED_FACTOR_MAX = 1.22       # slow-stroke swelling ceiling

# Corner pooling — the pen decelerates through a turn and ink accumulates.
CORNER_START_DEG = 28.0       # turn angle where pooling begins
CORNER_FULL_DEG = 100.0       # turn angle of maximum pooling
CORNER_POOL_GAIN = 0.30       # max extra width at a full corner

# Terminals
START_BLOB_GAIN = 0.20        # pen-down blob when the stroke starts slowly
END_POOL_GAIN = 0.16          # ink pool on a slow lift
FLICK_SPEED_RATIO = 1.45      # end speed ÷ median speed that counts as a flick
FLICK_TAPER_END = 0.22        # width fraction at the very tip of a flick tail
FLICK_LEN_FRAC = 2.6          # flick taper length, × pen_size

# Edge texture (applied per side, independently seeded)
EDGE_WAVE_AMP = 0.12          # waviness amplitude, × local half-width
EDGE_WAVE_LEN_FRAC = 3.0      # waviness wavelength, × pen_size
FIBER_NOTCH_THRESHOLD = 0.74  # noise below -this → paper-fibre notch
FIBER_NOTCH_DEPTH = 0.34      # notch depth, × local half-width
FIBER_WAVE_LEN_FRAC = 1.1     # notch noise wavelength, × pen_size

MIN_HALF_WIDTH = 0.35         # px — ink never thins to nothing mid-stroke
OUTLINE_DP_EPSILON = 0.18     # px — simplification before Bézier fitting
# Cubic→line conversion tolerance for the fitted outline. centerline.py's
# 0.8 px is tuned for plotter centerlines; on an ink EDGE that much
# flattening reads as polygon facets, so outlines keep their curves unless
# they are straight to within a tenth of a pixel.
LINEARIZE_EPSILON = 0.1
CAP_STEPS = 6                 # semicircular cap segments (30° each)


# ── Deterministic 1D value noise ────────────────────────────────────────────

def _seed_int(key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


def _hash1d(i: int, seed: int) -> float:
    h = (i ^ seed ^ 0x27D4EB2D) & 0xFFFFFFFF
    h = (h * 0x27D4EB2D) & 0xFFFFFFFF
    h ^= h >> 15
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h ^= h >> 13
    return ((h & 0xFFFF) / 32767.5) - 1.0


def _noise1d(x: float, seed: int) -> float:
    i = math.floor(x)
    f = x - i
    a = _hash1d(int(i), seed)
    b = _hash1d(int(i) + 1, seed)
    t = f * f * (3.0 - 2.0 * f)
    return a + t * (b - a)


# ── Small numeric helpers ───────────────────────────────────────────────────

def _smooth(values: List[float], window: int) -> List[float]:
    """Centered moving average; endpoints keep their raw values."""
    n = len(values)
    if n < 3 or window < 3:
        return list(values)
    half = window // 2
    out = list(values)
    for i in range(1, n - 1):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        out[i] = sum(values[lo:hi]) / (hi - lo)
    return out


def _median(values: List[float]) -> float:
    vals = sorted(v for v in values if v > 0)
    if not vals:
        return 0.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2.0


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ── Input parsing ───────────────────────────────────────────────────────────

def _parse_stroke(stroke: Sequence[Sequence[float]]):
    """
    Normalize one raw stroke into parallel lists (x, y, pressure, speed).

    Speed comes from timestamps when the points carry them ([x, y, p, t]);
    otherwise inter-sample distance is used as a relative proxy. Either way
    the ink model only ever uses speed RELATIVE to the stroke median, so the
    two sources behave consistently.
    """
    pts: List[Tuple[float, float, float, Optional[float]]] = []
    for raw in stroke:
        if len(raw) < 2:
            continue
        x, y = float(raw[0]), float(raw[1])
        p = _clamp(float(raw[2]), 0.05, 1.0) if len(raw) > 2 else 0.6
        t = float(raw[3]) if len(raw) > 3 else None
        if pts and math.hypot(x - pts[-1][0], y - pts[-1][1]) < 0.35:
            continue
        pts.append((x, y, p, t))
    if len(pts) < 2:
        return pts, [], [], []

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ps = [p[2] for p in pts]

    have_time = all(p[3] is not None for p in pts) and pts[-1][3] > pts[0][3]
    speeds = [0.0] * len(pts)
    for i in range(1, len(pts)):
        dist = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
        if have_time:
            dt = pts[i][3] - pts[i - 1][3]
            speeds[i] = dist / dt * 1000.0 if dt > 0 else speeds[i - 1]
        else:
            speeds[i] = dist
    speeds[0] = speeds[1]
    return pts, xs, ys, [ps, speeds]


def _resample(xs, ys, ps, vs, step: float):
    """Uniform arc-length resampling with linear interpolation of p and v."""
    arc = [0.0]
    for i in range(1, len(xs)):
        arc.append(arc[-1] + math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
    total = arc[-1]
    if total < step:
        return [xs[0], xs[-1]], [ys[0], ys[-1]], [ps[0], ps[-1]], [vs[0], vs[-1]], total

    n_out = max(3, int(total / step) + 1)
    rx, ry, rp, rv = [], [], [], []
    j = 0
    for k in range(n_out):
        s = total * k / (n_out - 1)
        while j < len(arc) - 2 and arc[j + 1] < s:
            j += 1
        seg = arc[j + 1] - arc[j]
        t = (s - arc[j]) / seg if seg > 1e-9 else 0.0
        rx.append(xs[j] + (xs[j + 1] - xs[j]) * t)
        ry.append(ys[j] + (ys[j + 1] - ys[j]) * t)
        rp.append(ps[j] + (ps[j + 1] - ps[j]) * t)
        rv.append(vs[j] + (vs[j + 1] - vs[j]) * t)
    return rx, ry, rp, rv, total


# ── Width model ─────────────────────────────────────────────────────────────

def _corner_pooling(xs, ys, step: float, pen_size: float) -> List[float]:
    """Per-point pooling factor (≥1) from local turn angle, spread outward
    with a triangular kernel so the swell fades in/out like wet ink."""
    n = len(xs)
    pool = [0.0] * n
    look = max(2, int(round(pen_size * 0.5 / step)))
    for i in range(look, n - look):
        v1x, v1y = xs[i] - xs[i - look], ys[i] - ys[i - look]
        v2x, v2y = xs[i + look] - xs[i], ys[i + look] - ys[i]
        n1 = math.hypot(v1x, v1y)
        n2 = math.hypot(v2x, v2y)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        dot = _clamp((v1x * v2x + v1y * v2y) / (n1 * n2), -1.0, 1.0)
        ang = math.degrees(math.acos(dot))
        pool[i] = _clamp((ang - CORNER_START_DEG) /
                         (CORNER_FULL_DEG - CORNER_START_DEG), 0.0, 1.0)

    radius = max(2, int(round(pen_size / step)))
    spread = [0.0] * n
    for i in range(n):
        if pool[i] <= 0:
            continue
        for d in range(-radius, radius + 1):
            j = i + d
            if 0 <= j < n:
                w = 1.0 - abs(d) / (radius + 1)
                if pool[i] * w > spread[j]:
                    spread[j] = pool[i] * w
    return [1.0 + CORNER_POOL_GAIN * s for s in spread]


def _terminal_envelope(n: int, arc_step: float, pen_size: float,
                       vs: List[float], v_med: float) -> List[float]:
    """
    Width envelope for stroke start/end.

    Start: quick ramp-in (the round cap hides the tip), plus a pen-down blob
    when the stroke begins slower than half the typical speed.
    End: a flick tail (long thin taper) when the pen lifted fast, otherwise a
    small ink pool where the pen stopped before lifting.
    """
    env = [1.0] * n
    if n < 3:
        return env

    ramp_pts = max(1, int(round(pen_size * 1.0 / arc_step)))
    for i in range(min(ramp_pts, n)):
        t = i / ramp_pts
        env[i] *= 0.70 + 0.30 * (1.0 - (1.0 - t) ** 2)

    slow_start = v_med > 0 and vs[0] < 0.5 * v_med
    if slow_start:
        blob_pts = max(2, int(round(pen_size * 1.8 / arc_step)))
        for i in range(min(blob_pts, n)):
            t = i / blob_pts
            env[i] *= 1.0 + START_BLOB_GAIN * (1.0 - t) ** 1.5

    fast_lift = v_med > 0 and vs[-1] > FLICK_SPEED_RATIO * v_med
    if fast_lift:
        flick_pts = max(3, int(round(pen_size * FLICK_LEN_FRAC / arc_step)))
        for i in range(min(flick_pts, n)):
            idx = n - 1 - i
            t = 1.0 - i / flick_pts          # 1 at the tip, 0 where taper starts
            env[idx] *= FLICK_TAPER_END + (1.0 - FLICK_TAPER_END) * (1.0 - t) ** 1.6
    else:
        pool_pts = max(2, int(round(pen_size * 1.0 / arc_step)))
        for i in range(min(pool_pts, n)):
            idx = n - 1 - i
            t = 1.0 - i / pool_pts
            env[idx] *= 1.0 + END_POOL_GAIN * t ** 1.5
    return env


# ── Outline assembly ────────────────────────────────────────────────────────

def _unit_normals(xs, ys) -> List[Point]:
    """Smoothed unit normals from central-difference tangents."""
    n = len(xs)
    tangents: List[Point] = []
    for i in range(n):
        a = max(0, i - 1)
        b = min(n - 1, i + 1)
        dx, dy = xs[b] - xs[a], ys[b] - ys[a]
        length = math.hypot(dx, dy)
        if length < 1e-9:
            tangents.append(tangents[-1] if tangents else (1.0, 0.0))
        else:
            tangents.append((dx / length, dy / length))
    smoothed: List[Point] = []
    for i in range(n):
        a = max(0, i - 1)
        b = min(n - 1, i + 1)
        tx = sum(t[0] for t in tangents[a:b + 1])
        ty = sum(t[1] for t in tangents[a:b + 1])
        length = math.hypot(tx, ty)
        if length < 1e-9:
            smoothed.append(tangents[i])
        else:
            smoothed.append((tx / length, ty / length))
    return [(-ty, tx) for tx, ty in smoothed]


def _cap_points(cx: float, cy: float, tx: float, ty: float,
                radius: float) -> List[Point]:
    """
    Semicircular tip cap from the +normal side around to the -normal side,
    bulging `radius` along the (unit) tangent direction (tx, ty).
    Returns the intermediate points only (endpoints are the side outlines).
    """
    nx, ny = -ty, tx
    pts: List[Point] = []
    for k in range(1, CAP_STEPS):
        ang = math.pi / 2 - math.pi * k / CAP_STEPS   # +90° → -90°
        ca, sa = math.cos(ang), math.sin(ang)
        # local frame: tangent = "forward", normal = "left"
        px = cx + radius * (ca * tx + sa * nx)
        py = cy + radius * (ca * ty + sa * ny)
        pts.append((px, py))
    return pts


def _simplify_gentle(start: Point, segs: List) -> List:
    """Cubic→line only when genuinely straight (LINEARIZE_EPSILON), then merge
    consecutive collinear lines. Same idea as centerline._simplify_segments
    but with a tolerance an order of magnitude tighter — outline curves must
    stay curves or the ink edge reads as polygon facets."""
    out: List = []
    prev = start
    for seg in segs:
        if seg[0] == 'cubic':
            _, c1, c2, p = seg
            if (_perp_distance(c1, prev, p) <= LINEARIZE_EPSILON and
                    _perp_distance(c2, prev, p) <= LINEARIZE_EPSILON):
                seg = ('line', p)
        if seg[0] == 'line' and out and out[-1][0] == 'line':
            # merge with the previous line when the shared point is collinear
            before = start if len(out) == 1 else out[-2][-1]
            if _perp_distance(out[-1][-1], before, seg[-1]) <= LINEARIZE_EPSILON:
                out[-1] = ('line', seg[-1])
                prev = seg[-1]
                continue
        out.append(seg)
        prev = seg[-1]
    return out


def _loop_to_svg(loop: List[Point]) -> str:
    """Simplify a closed point loop and fit smooth Béziers → SVG d string."""
    if len(loop) < 4:
        return ""
    pts = _douglas_peucker(loop, OUTLINE_DP_EPSILON)
    if len(pts) < 3:
        return ""
    runs = _split_at_corners(pts)
    segs = []
    for run in runs:
        segs.extend(_fit_run(run))
    segs = _simplify_gentle(pts[0], segs)
    if not segs:
        return ""
    parts = [f"M {pts[0][0]:.2f} {pts[0][1]:.2f}"]
    parts.extend(_segments_to_svg_parts(segs))
    parts.append("Z")
    return " ".join(parts)


def _ink_dot(cx: float, cy: float, radius: float, seed: int) -> str:
    """A small irregular ink blob — used for dot-like strokes (periods, the
    dot of an i, a quick tap). Wobbled circle fitted with smooth Béziers."""
    steps = 12
    loop: List[Point] = []
    for k in range(steps + 1):
        ang = 2 * math.pi * (k % steps) / steps
        r = radius * (1.0 + 0.12 * _noise1d(k * 0.9, seed))
        loop.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    runs = [loop]
    segs = []
    for run in runs:
        segs.extend(_fit_run(run))
    if not segs:
        return ""
    parts = [f"M {loop[0][0]:.2f} {loop[0][1]:.2f}"]
    parts.extend(_segments_to_svg_parts(segs))
    parts.append("Z")
    return " ".join(parts)


def _stroke_outline(stroke: Sequence[Sequence[float]], pen_size: float,
                    seed: int) -> str:
    """One pen stroke → one closed realistic-ink outline path (SVG d)."""
    pts, xs, ys, extra = _parse_stroke(stroke)
    base_hw = pen_size / 2.0

    if len(pts) < 2:
        # A tap so small it deduped to one point still leaves ink on paper.
        if pts:
            x, y, p, _ = pts[0]
            r = base_hw * (PRESSURE_BASE + PRESSURE_GAIN * p)
            return _ink_dot(x, y, max(r, 0.8), seed)
        return ""
    ps, vs = extra

    # Total raw length — degenerate strokes become ink dots.
    raw_len = sum(math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1])
                  for i in range(1, len(xs)))
    if raw_len < max(2.0, pen_size * 0.4):
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        avg_p = sum(ps) / len(ps)
        r = base_hw * (PRESSURE_BASE + PRESSURE_GAIN * avg_p) * 1.05
        return _ink_dot(cx, cy, max(r, 0.8), seed)

    step = _clamp(pen_size * RESAMPLE_STEP_FRAC,
                  RESAMPLE_STEP_MIN_PX, RESAMPLE_STEP_MAX_PX)
    xs, ys, ps, vs, total = _resample(xs, ys, ps, vs, step)
    n = len(xs)
    arc_step = total / (n - 1) if n > 1 else step

    win = max(3, int(round(pen_size / arc_step)) | 1)
    ps = _smooth(ps, win)
    vs = _smooth(vs, win)
    v_med = _median(vs)

    # ── Half-width profile: pressure × speed × corner pooling × terminals ──
    hws: List[float] = []
    for i in range(n):
        w = base_hw * (PRESSURE_BASE + PRESSURE_GAIN * ps[i])
        if v_med > 0:
            rel = _clamp(vs[i] / v_med, 0.25, 4.0)
            w *= _clamp(rel ** (-SPEED_EXPONENT),
                        SPEED_FACTOR_MIN, SPEED_FACTOR_MAX)
        hws.append(w)

    pooling = _corner_pooling(xs, ys, arc_step, pen_size)
    envelope = _terminal_envelope(n, arc_step, pen_size, vs, v_med)
    hws = [hws[i] * pooling[i] * envelope[i] for i in range(n)]
    hws = _smooth(hws, win)

    # ── Per-side edge texture: correlated waviness + paper-fibre notches ──
    wave_scale = 1.0 / (EDGE_WAVE_LEN_FRAC * pen_size)
    fiber_scale = 1.0 / (FIBER_WAVE_LEN_FRAC * pen_size)
    side_hw: List[List[float]] = [[], []]
    for side in range(2):
        wave_seed = seed * 2 + side
        fiber_seed = seed * 2 + side + 7919
        for i in range(n):
            s = i * arc_step
            w = hws[i] * (1.0 + EDGE_WAVE_AMP * _noise1d(s * wave_scale, wave_seed))
            fz = _noise1d(s * fiber_scale, fiber_seed)
            if fz < -FIBER_NOTCH_THRESHOLD:
                depth = (-fz - FIBER_NOTCH_THRESHOLD) / (1.0 - FIBER_NOTCH_THRESHOLD)
                w -= hws[i] * FIBER_NOTCH_DEPTH * depth
            side_hw[side].append(max(MIN_HALF_WIDTH, w))

    normals = _unit_normals(xs, ys)
    left = [(xs[i] + normals[i][0] * side_hw[0][i],
             ys[i] + normals[i][1] * side_hw[0][i]) for i in range(n)]
    right = [(xs[i] - normals[i][0] * side_hw[1][i],
              ys[i] - normals[i][1] * side_hw[1][i]) for i in range(n)]

    # Round tip caps, sized to the local (already tapered/pooled) width.
    # normals are the tangent rotated 90° CCW, so tangent = (n.y, -n.x);
    # the end cap bulges forward (+T), the start cap backward (-T).
    end_t = (normals[-1][1], -normals[-1][0])
    start_t = (-normals[0][1], normals[0][0])
    end_r = (side_hw[0][-1] + side_hw[1][-1]) / 2.0
    start_r = (side_hw[0][0] + side_hw[1][0]) / 2.0
    end_cap = _cap_points(xs[-1], ys[-1], end_t[0], end_t[1], end_r)
    start_cap = _cap_points(xs[0], ys[0], start_t[0], start_t[1], start_r)

    # Closed loop: left side out, around the end cap, right side back,
    # around the start cap (start_cap runs right→left because its tangent
    # is reversed). Repeat the first point so corner-splitting sees closure.
    loop = left + end_cap + list(reversed(right)) + start_cap + [left[0]]
    return _loop_to_svg(loop)


# ── Public API ──────────────────────────────────────────────────────────────

def realistic_glyph_outlines(
    pen_paths: Sequence[Sequence[Sequence[float]]],
    pen_size: float = 6.0,
    seed_key: str = "",
) -> List[str]:
    """
    Convert one glyph's raw pen strokes into realistic-ink outline paths.

    Returns a list of closed SVG path strings (one per stroke) in the same
    canvas coordinate space as the stroke points. Returns [] when nothing
    could be stroked — callers should fall back to the stored classic
    svg_paths in that case.
    """
    if not pen_paths:
        return []
    pen_size = max(1.5, float(pen_size or 6.0))
    base_seed = _seed_int(seed_key or "glyph")
    out: List[str] = []
    for idx, stroke in enumerate(pen_paths):
        if not stroke:
            continue
        d = _stroke_outline(stroke, pen_size, base_seed + idx * 101)
        if d:
            out.append(d)
    return out
