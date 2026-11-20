"""Which YOLO boxes the detector keeps: per-class confidence thresholds, open-set duplicate removal, and
physical checks against phantom static objects (:func:`stands_off_floor`, :func:`line_of_sight_clear`).

ROS-free, numpy only (``test/test_detection_filters.py``).

The detector runs class-aware NMS (boxes of different classes never suppress each other). The
old class-agnostic NMS at IoU 0.1 suppressed any box overlapping a higher-scoring one by 10 %:
two people side by side became one, and people in front of a large object vanished. Class-aware
NMS keeps them, but lets the open-set ``unknown`` class fire on what a known class already
covers, so :func:`drop_unknown_duplicates` removes an ``unknown`` box that

* overlaps a known-class box with IoU above ``dup_iou`` (the same thing, detected twice), or
* lies mostly (``inside_frac`` of its area) inside a PERSON box: part of the person (a bag, a
  tie, an arm), which must not become a static object.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def confident(conf: np.ndarray, clss: np.ndarray, *, unknown_class: int = 0, unknown_conf: float = 0.25,
              known_conf: float = 0.6, class_conf: Optional[Dict[int, float]] = None) -> np.ndarray:
    """``(N,)`` bool: detections at or above their class's threshold.

    ``unknown_class`` uses ``unknown_conf``; a class in ``class_conf`` uses its own threshold
    (e.g. ``{person: 0.4}``); every other known class uses ``known_conf``.
    """
    conf = np.asarray(conf, dtype=np.float64).reshape(-1)
    clss = np.asarray(clss).reshape(-1)
    th = np.full(len(conf), float(known_conf))
    th[clss == unknown_class] = float(unknown_conf)
    for c, t in (class_conf or {}).items():
        th[clss == c] = float(t)
    return conf >= th


def _area(b: np.ndarray) -> np.ndarray:
    return np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)


def _intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(len(a), len(b))`` intersection areas of xyxy boxes."""
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return ix * iy


def drop_unknown_duplicates(boxes: np.ndarray, clss: np.ndarray, keep: np.ndarray, *, unknown_class: int = 0,
                            person_classes: Sequence[int] = (), dup_iou: float = 0.5,
                            inside_frac: float = 0.8) -> np.ndarray:
    """``keep`` with the kept ``unknown`` boxes removed that duplicate a kept known-class box (see the
    module doc). Only boxes already kept count on either side."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    clss = np.asarray(clss).reshape(-1)
    keep = np.asarray(keep, dtype=bool).copy()
    unk = np.where(keep & (clss == unknown_class))[0]
    known = np.where(keep & (clss != unknown_class))[0]
    if not len(unk) or not len(known):
        return keep
    inter = _intersection(boxes[unk], boxes[known])
    a_unk, a_known = _area(boxes[unk]), _area(boxes[known])
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / (a_unk[:, None] + a_known[None, :] - inter)
        inside = inter / a_unk[:, None]
    iou, inside = np.nan_to_num(iou), np.nan_to_num(inside)
    is_person = np.isin(clss[known], np.asarray(list(person_classes), dtype=clss.dtype))
    dup = (iou > dup_iou).any(axis=1) | ((inside >= inside_frac) & is_person[None, :]).any(axis=1)
    keep[unk[dup]] = False
    return keep


# --------------------------------------------------------------------------- phantom static objects
#
# Two physical checks for non-person detections, so a recurring false box (a rug, a floor pattern or
# shadow, a reflection in glass) is not reported as an obstacle:
#
# * :func:`stands_off_floor`: part of the object must rise ``min_height`` above the floor. The box's depth
#   pixels near the object's own depth are back-projected through the camera (its height above the floor
#   and its pitch); anything flat on the floor fails.
# * :func:`line_of_sight_clear`: the camera must be able to see the object: the ray from the camera to
#   its near surface must not cross a wall of the static map (a reflection lands behind the glass).


def heights_above_floor(depth_mm: np.ndarray, x1: int, y1: int, x2: int, y2: int, depth_m: float, fy: float,
                        cy: float, cam_height: float, pitch: float = 0.0, stride: int = 4,
                        band: float = 0.5) -> np.ndarray:
    """Heights (m above the floor) of the box's depth pixels within ``band`` of the object's depth ``depth_m``.

    ``depth_mm`` is the (upright, i.e. already rotated for an upside-down mount) depth image in mm; ``cy``
    the principal point row in that image; ``cam_height`` the optical centre's height above the floor;
    ``pitch`` the camera's downward tilt (rad, positive = looking down). Only pixels near the object's depth
    count, so background and the floor around it do not."""
    h, w = depth_mm.shape[:2]
    x1, x2 = max(0, int(x1)), min(w, int(x2))
    y1, y2 = max(0, int(y1)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros(0)
    win = depth_mm[y1:y2:stride, x1:x2:stride].astype(np.float64) / 1000.0
    rows = np.arange(y1, y2, stride, dtype=np.float64)[:, None] * np.ones((1, win.shape[1]))
    near = (win > 0.0) & (np.abs(win - float(depth_m)) <= max(float(band), 0.1 * float(depth_m)))
    z, v = win[near], rows[near]
    y_down = (v - float(cy)) * z / float(fy)  # optical frame: +y down, +z forward
    return float(cam_height) - (y_down * np.cos(pitch) + z * np.sin(pitch))


def stands_off_floor(heights: np.ndarray, min_height: float = 0.15, min_frac: float = 0.10,
                     min_pixels: int = 20) -> Optional[bool]:
    """True if at least ``min_frac`` of the object's pixels are ``min_height`` or more above the floor;
    None when there are too few pixels to judge (the caller keeps the detection)."""
    hgt = np.asarray(heights, dtype=np.float64).reshape(-1)
    if len(hgt) < int(min_pixels):
        return None
    return bool(np.mean(hgt >= float(min_height)) >= float(min_frac))


def line_of_sight_clear(occupied: np.ndarray, resolution: float, origin: Sequence[float], start: Sequence[float],
                        end: Sequence[float], skip: float = 0.25) -> bool:
    """False if the segment from ``start`` (the camera) to ``end`` (the object's near surface), both in the
    map frame, crosses an occupied cell of ``occupied`` ((H, W) bool, row = y). The first ``skip`` metres
    are ignored (the robot's own cell). Cells off the map count as free."""
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    length = float(np.hypot(ex - sx, ey - sy))
    if length <= skip:
        return True
    n = int(np.ceil((length - skip) / (0.5 * resolution))) + 1
    s = np.linspace(skip, length, n) / length
    px, py = sx + s * (ex - sx), sy + s * (ey - sy)
    ix = np.floor((px - float(origin[0])) / resolution).astype(np.int64)
    iy = np.floor((py - float(origin[1])) / resolution).astype(np.int64)
    hgt, wid = occupied.shape
    on = (ix >= 0) & (ix < wid) & (iy >= 0) & (iy < hgt)
    return not bool(occupied[iy[on], ix[on]].any())
