"""Which YOLO boxes the detector keeps: per-class confidence thresholds and open-set duplicate removal.

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
