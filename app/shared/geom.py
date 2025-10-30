import numpy as np
import shapely.geometry as sgeom
import alphashape
from .data import MAX_POINTS_FOR_ENVELOPE, MAX_POINTS_FOR_KNN, KNN_K

def _sample_points(arr, max_n):
    n = len(arr)
    if n <= max_n:
        return arr
    idx = np.linspace(0, n - 1, num=max_n, dtype=int)
    return arr[idx]

def _estimate_alpha(pts, k=KNN_K):
    pts = _sample_points(pts, MAX_POINTS_FOR_KNN)
    if len(pts) < k + 2:
        return 1.0
    A = pts[:, None, :] - pts[None, :, :]
    D = np.sqrt((A**2).sum(axis=2))
    D.sort(axis=1)
    dk = np.median(D[:, k])
    if not np.isfinite(dk) or dk <= 1e-9:
        return 1.0
    return 1.0 / (1.8 * dk)

def compute_boundary_curve(xy_points, prefer_concave=True):
    pts = np.asarray(xy_points, float)
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) < 3:
        return None
    pts = np.unique(pts, axis=0)
    if len(pts) < 3:
        return None

    poly = None
    if prefer_concave:
        try:
            pts_cap = _sample_points(pts, MAX_POINTS_FOR_ENVELOPE)
            alpha = _estimate_alpha(pts_cap)
            poly = alphashape.alphashape(pts_cap, alpha)
        except Exception:
            poly = None

    if poly is None:
        poly = sgeom.MultiPoint(pts).convex_hull

    if isinstance(poly, sgeom.MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)

    if isinstance(poly, sgeom.Polygon):
        x, y = poly.exterior.coords.xy
        return np.column_stack([x, y])
    if isinstance(poly, sgeom.LineString):
        x, y = poly.coords.xy
        return np.column_stack([x, y])

    return None
