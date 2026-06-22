"""Convert a 1-pixel-wide skeleton image into ordered polylines.

The skeleton (a boolean array where True marks centerline pixels) is treated
as an 8-connected graph. Pixels are classified by their neighbour count:

    * degree 1        -> endpoint   (a line tip)
    * degree 2        -> path pixel (the body of a stroke)
    * degree 3 or 4+  -> junction   (where strokes meet/cross)

``skeleton_to_polylines`` walks every edge of that graph once, cutting a new
polyline at every junction and endpoint. ``skeleton_to_strokes`` then links
those edges back together through junctions using tangent continuity, so a line
that crosses or touches another line stays a single stroke that *bends* through
the junction instead of fragmenting into many pieces.
"""

from __future__ import annotations

import math

import numpy as np

# 8-connectivity neighbour offsets.
_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

# Edges that loop within a single junction cluster and are no longer than this
# are treated as skeletonisation artifacts and discarded. Real loops are longer.
_ARTIFACT_LEN = 6


def _build_adjacency(points):
    """Map each skeleton pixel to its list of 8-connected skeleton neighbours."""
    adj = {}
    for r, c in points:
        nbrs = []
        for dr, dc in _OFFSETS:
            p = (r + dr, c + dc)
            if p in points:
                nbrs.append(p)
        adj[(r, c)] = nbrs
    return adj


def _edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def _trace_edges(points, adj, degree):
    """Split the skeleton graph into edges between nodes, plus isolated loops.

    Returns ``(edges, loops, node_pixels)`` where each edge is a dict with a
    ``poly`` (list of ``(row, col)`` pixels from one node to the next) and its
    two endpoint pixels ``a`` / ``b``. ``loops`` are closed cycles with no node.
    """
    node_pixels = {p for p in points if degree[p] != 2}
    visited = set()

    def walk(start, first):
        poly = [start, first]
        visited.add(_edge_key(start, first))
        prev, cur = start, first
        while cur not in node_pixels:
            nxt = None
            for cand in adj[cur]:
                if cand == prev or _edge_key(cur, cand) in visited:
                    continue
                nxt = cand
                break
            if nxt is None:
                break
            visited.add(_edge_key(cur, nxt))
            poly.append(nxt)
            prev, cur = cur, nxt
        return poly

    edges = []
    for n in node_pixels:
        for nb in adj[n]:
            if _edge_key(n, nb) not in visited:
                poly = walk(n, nb)
                edges.append({"poly": poly, "a": poly[0], "b": poly[-1]})

    loops = []
    for p in points:
        for nb in adj[p]:
            if _edge_key(p, nb) in visited:
                continue
            poly = [p, nb]
            visited.add(_edge_key(p, nb))
            prev, cur = p, nb
            while cur != p:
                nxt = None
                for cand in adj[cur]:
                    if cand == prev or _edge_key(cur, cand) in visited:
                        continue
                    nxt = cand
                    break
                if nxt is None:
                    break
                visited.add(_edge_key(cur, nxt))
                poly.append(nxt)
                prev, cur = cur, nxt
            loops.append(poly)

    return edges, loops, node_pixels


def skeleton_to_polylines(skel: np.ndarray):
    """Trace ``skel`` into raw polylines, one per graph edge (no linking)."""
    coords = np.argwhere(skel)
    if coords.size == 0:
        return []
    points = {(int(r), int(c)) for r, c in coords}
    adj = _build_adjacency(points)
    degree = {p: len(nbrs) for p, nbrs in adj.items()}
    edges, loops, _ = _trace_edges(points, adj, degree)
    return [e["poly"] for e in edges] + loops


# --------------------------------------------------------------------------- #
# Stroke linking
# --------------------------------------------------------------------------- #

def _cluster_nodes(node_pixels):
    """Group touching node pixels (8-connected) into single logical nodes."""
    node_set = set(node_pixels)
    cluster_of = {}
    cid = 0
    for p in node_set:
        if p in cluster_of:
            continue
        stack = [p]
        cluster_of[p] = cid
        while stack:
            q = stack.pop()
            for dr, dc in _OFFSETS:
                n = (q[0] + dr, q[1] + dc)
                if n in node_set and n not in cluster_of:
                    cluster_of[n] = cid
                    stack.append(n)
        cid += 1
    return cluster_of


def _stub_direction(poly, end, span=6):
    """Unit vector pointing away from the node at ``end`` ('a' or 'b')."""
    n = len(poly)
    if end == "a":
        p0 = poly[0]
        p1 = poly[min(span, n - 1)]
    else:
        p0 = poly[-1]
        p1 = poly[max(0, n - 1 - span)]
    vr, vc = p1[0] - p0[0], p1[1] - p0[1]
    mag = math.hypot(vr, vc) or 1.0
    return (vr / mag, vc / mag)


def _bend_degrees(d1, d2):
    """Turn angle (deg) for a stroke passing from one stub to another.

    Both stubs point *away* from the shared node, so a perfectly straight
    pass-through has opposite directions (dot = -1) and returns 0 degrees.
    """
    dot = max(-1.0, min(1.0, d1[0] * d2[0] + d1[1] * d2[1]))
    return math.degrees(math.pi - math.acos(dot))


def skeleton_to_strokes(
    skel: np.ndarray,
    max_bend: float = 75.0,
    spur_length: int = 6,
    tangent_span: int = 6,
):
    """Trace ``skel`` into a minimal set of strokes by linking through junctions.

    ``max_bend``     -- largest turn angle (deg) still treated as the same line
                        continuing through a junction. Higher merges more.
    ``spur_length``  -- prune dead-end barbs shorter than this many pixels.
    ``tangent_span`` -- how many pixels from a node to measure its direction.
    """
    coords = np.argwhere(skel)
    if coords.size == 0:
        return []
    points = {(int(r), int(c)) for r, c in coords}
    adj = _build_adjacency(points)
    degree = {p: len(nbrs) for p, nbrs in adj.items()}

    edges, loops, node_pixels = _trace_edges(points, adj, degree)
    if not edges:
        return loops

    cluster_of = _cluster_nodes(node_pixels)
    next_cid = (max(cluster_of.values()) + 1) if cluster_of else 0

    def cluster_id(pixel):
        nonlocal next_cid
        if pixel not in cluster_of:
            cluster_of[pixel] = next_cid
            next_cid += 1
        return cluster_of[pixel]

    for e in edges:
        e["ca"] = cluster_id(e["a"])
        e["cb"] = cluster_id(e["b"])
        e["alive"] = True

    # --- Prune short dead-end spurs (barbs off a junction). ---------------- #
    if spur_length > 0:
        cluster_degree = {}
        for e in edges:
            cluster_degree[e["ca"]] = cluster_degree.get(e["ca"], 0) + 1
            cluster_degree[e["cb"]] = cluster_degree.get(e["cb"], 0) + 1
        for e in edges:
            if len(e["poly"]) > spur_length:
                continue
            deg_a, deg_b = cluster_degree[e["ca"]], cluster_degree[e["cb"]]
            # A spur dangles from a junction (deg>=3) to a free tip (deg==1).
            if (deg_a == 1 and deg_b >= 3) or (deg_b == 1 and deg_a >= 3):
                e["alive"] = False

    # --- Drop tiny artifact edges that start and end in the same junction. -- #
    # An 8-connected crossing collapses into a clique of junction pixels; the
    # short edges looping within that clique are noise, not real strokes.
    for e in edges:
        if e["alive"] and e["ca"] == e["cb"] and len(e["poly"]) <= _ARTIFACT_LEN:
            e["alive"] = False

    live = [e for e in edges if e["alive"]]
    if not live:
        return loops

    # --- Collect the two stubs of every edge at their clusters. ------------ #
    stubs_at = {}  # cluster_id -> list of (edge_index, end)
    for idx, e in enumerate(live):
        d_a = _stub_direction(e["poly"], "a", tangent_span)
        d_b = _stub_direction(e["poly"], "b", tangent_span)
        e["dir"] = {"a": d_a, "b": d_b}
        stubs_at.setdefault(e["ca"], []).append((idx, "a"))
        stubs_at.setdefault(e["cb"], []).append((idx, "b"))

    # --- Greedily pair the straightest stubs at each junction. ------------- #
    partner = {}  # (edge_index, end) -> (edge_index, end)
    for cid, stubs in stubs_at.items():
        if len(stubs) < 2:
            continue
        candidates = []
        for i in range(len(stubs)):
            for j in range(i + 1, len(stubs)):
                ei, endi = stubs[i]
                ej, endj = stubs[j]
                if ei == ej:
                    continue  # don't link an edge to itself at one node
                bend = _bend_degrees(live[ei]["dir"][endi], live[ej]["dir"][endj])
                candidates.append((bend, i, j))
        candidates.sort(key=lambda t: t[0])
        used = set()
        for bend, i, j in candidates:
            if bend > max_bend:
                break
            if i in used or j in used:
                continue
            used.add(i)
            used.add(j)
            si, sj = stubs[i], stubs[j]
            partner[si] = sj
            partner[sj] = si

    # --- Walk linked chains into continuous strokes. ---------------------- #
    other_end = {"a": "b", "b": "a"}
    visited = set()
    strokes = []

    def oriented(idx, entry_end):
        poly = live[idx]["poly"]
        return poly if entry_end == "a" else poly[::-1]

    def build_chain(start_idx, entry_end):
        chain = []
        idx, end = start_idx, entry_end
        while idx is not None and idx not in visited:
            visited.add(idx)
            seg = oriented(idx, end)
            if chain and seg and chain[-1] == seg[0]:
                seg = seg[1:]
            chain.extend(seg)
            exit_end = other_end[end]
            nxt = partner.get((idx, exit_end))
            if nxt is None:
                break
            idx, end = nxt
        return chain

    # Start chains at open ends (a stub with no partner), so each chain is
    # built once from one of its two free tips.
    for idx, e in enumerate(live):
        for end in ("a", "b"):
            if (idx, end) not in partner and idx not in visited:
                strokes.append(build_chain(idx, end))

    # Anything still unvisited is a closed ring of linked edges.
    for idx in range(len(live)):
        if idx not in visited:
            strokes.append(build_chain(idx, "a"))

    return [s for s in strokes if len(s) >= 2] + loops
