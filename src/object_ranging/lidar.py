"""Range a detection with the 2D LiDAR: the scan returns inside the box's bearing span."""

import math

import numpy as np


def scan_points_map(ranges, angles, laser_x, laser_y, laser_yaw, range_min=0.0, range_max=np.inf):
    """``(N, 2)`` map-frame points of the valid returns (finite, within [range_min, range_max])."""
    r = np.asarray(ranges, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)
    ok = np.isfinite(r) & (r >= max(range_min, 1e-3)) & (r <= range_max)
    r, a = r[ok], a[ok] + laser_yaw
    return np.stack([laser_x + r * np.cos(a), laser_y + r * np.sin(a)], axis=1)


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def range_in_bearing_span(points_map, x_camera, y_camera, theta_camera, bearing_lo, bearing_hi, *,
                          margin=0.15, cluster_gap=0.25, min_points=3, max_range=8.0, prefer_range=None,
                          gate=0.7):
    """The LiDAR cluster the box is looking at, or None.

    Keeps the points whose bearing from the CAMERA (so the camera/laser offset is handled) is
    inside ``[bearing_lo, bearing_hi]`` shrunk by ``margin`` of its width on each side (keeps
    neighbours out), within ``max_range``; splits them into clusters along range wherever
    consecutive ranges jump more than ``cluster_gap``; keeps clusters of ``min_points`` or more.
    Returns the cluster nearest to ``prefer_range`` (within ``gate``) when it is given -- the
    camera depth, so a chair in front of a person at scan height does not win -- otherwise the
    nearest. Result: ``(x, y, range)``, the cluster's map-frame centroid and median range.
    """
    lo, hi = min(bearing_lo, bearing_hi), max(bearing_lo, bearing_hi)
    shrink = margin * (hi - lo)
    lo, hi = lo + shrink, hi - shrink
    p = np.asarray(points_map, dtype=np.float64).reshape(-1, 2)
    if len(p) == 0:
        return None
    d = p - np.array([x_camera, y_camera])
    rng = np.hypot(d[:, 0], d[:, 1])
    brg = _wrap(np.arctan2(d[:, 1], d[:, 0]) - theta_camera)
    keep = (brg >= lo) & (brg <= hi) & (rng <= max_range) & (rng > 1e-3)
    if np.count_nonzero(keep) < min_points:
        return None
    p, rng = p[keep], rng[keep]
    order = np.argsort(rng)
    p, rng = p[order], rng[order]
    splits = np.nonzero(np.diff(rng) > cluster_gap)[0] + 1
    clusters = [(cp, cr) for cp, cr in zip(np.split(p, splits), np.split(rng, splits)) if len(cr) >= min_points]
    if not clusters:
        return None
    if prefer_range is not None and prefer_range > 0.0:
        best = min(clusters, key=lambda c: abs(float(np.median(c[1])) - prefer_range))
        if abs(float(np.median(best[1])) - prefer_range) > gate:
            best = clusters[0]  # nothing agrees with the camera: the nearest (the safe side)
    else:
        best = clusters[0]
    cp, cr = best
    return float(cp[:, 0].mean()), float(cp[:, 1].mean()), float(np.median(cr))
