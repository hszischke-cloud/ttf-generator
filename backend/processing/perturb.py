"""
perturb.py — Organic micro-serrations for glyph outlines.

Applies arc-length-parameterised 1D Perlin noise perpendicular to each
closed contour, simulating the natural edge irregularities of real
handwriting: small correlated divots that ebb and flow like a genuine
pen or pencil moving across paper tooth.

All coordinates are in font UPM space (1000-unit system) so the
amplitude parameter is consistent across different glyph sizes and
upscale factors regardless of the source SVG resolution.

Pen operation format (same as fontTools AbstractPen):
  ('moveTo',  (x, y))
  ('lineTo',  (x, y))
  ('curveTo', (cx1,cy1), (cx2,cy2), (x,y))   — cubic Bezier
  ('closePath',)
  ('endPath',)
"""

import math
from typing import List, Tuple

# Pen operation type alias
Op = tuple
Contour = List[Op]
Pt = Tuple[int, int]


# ---------------------------------------------------------------------------
# 1D Perlin gradient noise — no external dependencies
# ---------------------------------------------------------------------------

def _build_perm(seed: int) -> List[int]:
    """
    Build a 512-element permutation table seeded deterministically.
    Doubled so index lookups never wrap beyond the array length.
    """
    import random as _r
    rng = _r.Random(seed)
    p = list(range(256))
    rng.shuffle(p)
    return p + p


def _perlin1d(x: float, perm: List[int]) -> float:
    """
    1D Perlin gradient noise. Returns a value in [-1, +1].

    Uses the canonical quintic fade curve (6t⁵ - 15t⁴ + 10t³) for
    C² continuity at integer boundaries, giving smooth, non-jittery bumps
    that feel organic rather than randomly noisy.

    The raw Perlin output for the ±1 gradient scheme peaks at ±0.5
    (gradients average out at the midpoint between lattice nodes), so
    we multiply by 2 to stretch the output to the full [-1, +1] range.
    """
    xi = int(math.floor(x)) & 255
    xf = x - math.floor(x)
    # Quintic fade
    fade = xf * xf * xf * (xf * (xf * 6.0 - 15.0) + 10.0)
    # Gradient selection: hash → ±x
    ga = xf       if (perm[xi]     & 1) == 0 else -xf
    gb = (xf - 1) if (perm[xi + 1] & 1) == 0 else -(xf - 1)
    return (ga + fade * (gb - ga)) * 2.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _endpoint(op: Op) -> Pt:
    """Extract the on-curve endpoint from a moveTo / lineTo / curveTo op."""
    # curveTo: ('curveTo', c1, c2, end)  — last element is the on-curve point
    # moveTo / lineTo: ('moveTo'|'lineTo', pt)
    return op[-1]


def _on_curve_indices(contour: Contour) -> List[int]:
    """
    Return the indices within *contour* of operations that carry an
    on-curve endpoint (moveTo, lineTo, curveTo).
    """
    return [i for i, op in enumerate(contour)
            if op[0] in ('moveTo', 'lineTo', 'curveTo')]


def _arc_lengths(pts: List[Pt]) -> List[float]:
    """Cumulative straight-line arc length for a sequence of points."""
    lengths = [0.0]
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        lengths.append(lengths[-1] + math.hypot(dx, dy))
    return lengths


def _unit_tangent(pts: List[Pt], i: int, closed: bool) -> Tuple[float, float]:
    """
    Unit tangent at index *i* using central differences.
    Wraps around for closed contours; clamps for open ones.
    """
    n = len(pts)
    if closed:
        prev = pts[(i - 1) % n]
        nxt  = pts[(i + 1) % n]
    else:
        prev = pts[max(0, i - 1)]
        nxt  = pts[min(n - 1, i + 1)]
    dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return (1.0, 0.0)
    return (dx / length, dy / length)


# ---------------------------------------------------------------------------
# Core perturbation
# ---------------------------------------------------------------------------

def perturb_contour(
    contour: Contour,
    perm: List[int],
    amplitude: float,
    frequency: float,
    phase: float,
) -> Contour:
    """
    Perturb one closed/open subpath with correlated perpendicular noise.

    Strategy
    --------
    1. Extract the sequence of on-curve points (M, L, C endpoints).
    2. Compute cumulative arc length between them.
    3. At each on-curve point, sample Perlin noise at that arc-length
       position.  The noise drives a displacement *perpendicular* to the
       local stroke tangent — this produces divots along the edge rather
       than oscillation of the stroke width.
    4. For cubic Bezier control points (c1, c2):
       • c1 inherits the displacement of the *previous* on-curve point
         (it controls the exit from that point).
       • c2 inherits the displacement of the *current* on-curve endpoint
         (it controls the approach to that point).
       This keeps each Bezier segment's local curvature shape intact while
       shifting the segment body with the noise.

    Args:
        contour:   list of pen ops for one subpath
        perm:      Perlin permutation table, seeded per glyph
        amplitude: max edge displacement in font UPM units
        frequency: noise frequency in bumps per UPM of arc length
        phase:     noise phase offset — unique per contour per glyph
    """
    idx = _on_curve_indices(contour)
    if len(idx) < 2:
        return contour

    on_pts = [_endpoint(contour[i]) for i in idx]
    closed = any(op[0] == 'closePath' for op in contour)

    arc = _arc_lengths(on_pts)
    if arc[-1] < 1.0:
        return contour

    n = len(on_pts)

    # --- Compute displacement vector for each on-curve point ---
    deltas: List[Tuple[float, float]] = []
    for k in range(n):
        s = arc[k]
        noise = _perlin1d(s * frequency + phase, perm)
        tx, ty = _unit_tangent(on_pts, k, closed)
        # Normal = 90° CCW rotation of tangent
        nx, ny = -ty, tx
        deltas.append((nx * noise * amplitude, ny * noise * amplitude))

    # --- Rebuild contour with perturbed coordinates ---
    op_to_k = {op_i: k for k, op_i in enumerate(idx)}
    new_ops: List[Op] = []

    for i, op in enumerate(contour):
        name = op[0]

        if name == 'moveTo':
            k = op_to_k[i]
            dx, dy = deltas[k]
            x, y = on_pts[k]
            new_ops.append(('moveTo', (round(x + dx), round(y + dy))))

        elif name == 'lineTo':
            k = op_to_k[i]
            dx, dy = deltas[k]
            x, y = on_pts[k]
            new_ops.append(('lineTo', (round(x + dx), round(y + dy))))

        elif name == 'curveTo':
            # op = ('curveTo', c1, c2, end_pt)
            k = op_to_k[i]
            prev_k = (k - 1) % n if closed else max(0, k - 1)

            c1, c2, end_pt = op[1], op[2], op[3]
            pdx, pdy = deltas[prev_k]   # c1: exit from previous on-curve
            cdx, cdy = deltas[k]        # c2 + endpoint: approach to current

            new_ops.append(('curveTo',
                (round(c1[0] + pdx),    round(c1[1] + pdy)),
                (round(c2[0] + cdx),    round(c2[1] + cdy)),
                (round(end_pt[0] + cdx), round(end_pt[1] + cdy)),
            ))

        else:
            # closePath, endPath — no coordinates to adjust
            new_ops.append(op)

    return new_ops


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _glyph_seed(name: str) -> int:
    """
    Deterministic integer seed derived from a glyph name.

    Uses the first 3 bytes of MD5 so that adjacent characters ('a', 'b',
    'c' …) produce completely uncorrelated seeds — djb2 gives consecutive
    seeds for consecutive chars which makes their noise patterns nearly
    identical. Python's built-in hash() is randomised per process; MD5
    is always stable.
    """
    import hashlib
    return int(hashlib.md5(name.encode()).hexdigest()[:6], 16)


def perturb_glyph(
    contours: List[Contour],
    glyph_name: str,
    amplitude: float = 10.0,
    frequency: float = 0.012,
) -> List[Contour]:
    """
    Apply organic micro-serrations to all contours of one glyph.

    Each contour gets a distinct phase offset so the outer and inner edges
    of a stroke don't mirror each other — they evolve independently, which
    is what real handwriting looks like.

    Args:
        contours:   all subpaths of the glyph (list of contour op-lists)
        glyph_name: used to seed the noise so each letter has its own
                    unique but reproducible irregularity pattern
        amplitude:  max perpendicular edge displacement in UPM units.
                    Default 10 UPM ≈ 1.4 % of cap-height; visible at
                    display sizes (36 pt+) without looking broken.
        frequency:  Perlin noise frequency in bumps per UPM arc length.
                    Default 0.012 → ~7 bumps per 600 UPM stroke perimeter.
    """
    seed = _glyph_seed(glyph_name)
    perm = _build_perm(seed)
    result = []
    for j, contour in enumerate(contours):
        # Phase shifts: prime-multiple spacing keeps contours uncorrelated
        phase = j * 17.3 + (seed % 1000) * 0.137
        result.append(perturb_contour(contour, perm, amplitude, frequency, phase))
    return result
