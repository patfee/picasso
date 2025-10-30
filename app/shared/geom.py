# app/shared/geom.py
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def _to_array(xy):
    pts = np.asarray(xy, dtype=float)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts

def compute_boundary_curve(xy, mode="convex", alpha=1.0):
    """
    Returns an ordered boundary polyline for the given 2D points.
    mode: "none" | "convex" | "concave"
    alpha: controls concavity (higher => more concave). Simple alpha-shape style.
    """
    pts = _to_array(xy)
    if pts.shape[0] < 3:
        return pts
    mode = (mode or "convex").lower()

    if mode == "none":
        # No boundary; caller can just plot points
        return np.empty((0, 2))

    if mode == "convex":
        hull = ConvexHull(pts)
        return pts[hull.vertices]

    if mode == "concave":
        # Simple alpha-shape approximation using Delaunay & circumradius threshold
        tri = Delaunay(pts)
        simplices = tri.simplices
        T = pts[simplices]  # (n_tri, 3, 2)

        def circumradius(a, b, c):
            ab = np.linalg.norm(a - b)
            bc = np.linalg.norm(b - c)
            ca = np.linalg.norm(c - a)
            s = (ab + bc + ca) / 2.0
            area_sq = max(s * (s - ab) * (s - bc) * (s - ca), 0.0)
            if area_sq == 0:
                return np.inf
            area = area_sq ** 0.5
            return (ab * bc * ca) / (4.0 * area)

        keep = []
        for a, b, c in T:
            R = circumradius(a, b, c)
            if R < 1.0 / max(alpha, 1e-6):
                keep += [tuple(sorted((tuple(a), tuple(b)))),
                         tuple(sorted((tuple(b), tuple(c)))),
                         tuple(sorted((tuple(c), tuple(a))))]

        keep = list(set(keep))
        if not keep:
            hull = ConvexHull(pts)
            return pts[hull.vertices]

        from collections import defaultdict
        adj = defaultdict(list)
        for p, q in keep:
            adj[p].append(q)
            adj[q].append(p)

        # Walk a loop (best-effort). If not closed, just return the visited chain.
        start = next(iter(adj))
        boundary = [start]
        prev = None
        curr = start
        guard = 0
        while guard < len(keep) + 5:
            guard += 1
            nbrs = adj[curr]
            nxt = None
            for v in nbrs:
                if v != prev:
                    nxt = v
                    break
            if nxt is None:
                break
            boundary.append(nxt)
            prev, curr = curr, nxt
            if nxt == start:
                break

        return np.array(boundary, dtype=float)

    # Fallback = convex
    hull = ConvexHull(pts)
    return pts[hull.vertices]
