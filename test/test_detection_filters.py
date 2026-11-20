"""Unit tests for detection_filters (per-class thresholds, open-set duplicate removal)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detection_filters import confident, drop_unknown_duplicates  # noqa: E402

UNKNOWN, PERSON, CHAIR = 0, 1, 5


def test_per_class_thresholds():
    conf = np.array([0.30, 0.45, 0.45, 0.55, 0.65, 0.20])
    clss = np.array([UNKNOWN, PERSON, CHAIR, PERSON, CHAIR, UNKNOWN])
    np.testing.assert_array_equal(confident(conf, clss, class_conf={PERSON: 0.4}),
                                  [True, True, False, True, True, False])
    # without the person override, people need the known-class 0.6 (the old behaviour)
    np.testing.assert_array_equal(confident(conf, clss), [True, False, False, False, True, False])


def test_an_unknown_box_on_a_person_is_dropped():
    """The tie in ultralytics' zidane.jpg: an unknown box entirely inside a person box."""
    boxes = np.array([[60, 100, 549, 357], [221, 218, 264, 357]], dtype=float)
    clss = np.array([PERSON, UNKNOWN])
    keep = drop_unknown_duplicates(boxes, clss, np.ones(2, bool), person_classes=[PERSON])
    np.testing.assert_array_equal(keep, [True, False])


def test_an_unknown_duplicate_of_a_known_object_is_dropped():
    boxes = np.array([[100, 100, 200, 200], [105, 102, 205, 198]], dtype=float)
    clss = np.array([CHAIR, UNKNOWN])
    keep = drop_unknown_duplicates(boxes, clss, np.ones(2, bool), person_classes=[PERSON])
    np.testing.assert_array_equal(keep, [True, False])


def test_a_large_unknown_object_behind_people_is_kept():
    """The bus in ultralytics' bus.jpg: people overlap it a little; both stay."""
    boxes = np.array([[3, 136, 478, 440], [28, 237, 141, 536], [397, 217, 480, 520]], dtype=float)
    clss = np.array([UNKNOWN, PERSON, PERSON])
    keep = drop_unknown_duplicates(boxes, clss, np.ones(3, bool), person_classes=[PERSON])
    np.testing.assert_array_equal(keep, [True, True, True])


def test_an_unknown_inside_a_known_non_person_box_is_kept():
    """Containment only counts for people (a box on a table is a separate object)."""
    boxes = np.array([[0, 0, 300, 300], [100, 100, 140, 140]], dtype=float)
    clss = np.array([CHAIR, UNKNOWN])
    keep = drop_unknown_duplicates(boxes, clss, np.ones(2, bool), person_classes=[PERSON])
    np.testing.assert_array_equal(keep, [True, True])


def test_only_kept_boxes_count():
    boxes = np.array([[60, 100, 549, 357], [221, 218, 264, 357]], dtype=float)
    clss = np.array([PERSON, UNKNOWN])
    keep = drop_unknown_duplicates(boxes, clss, np.array([False, True]), person_classes=[PERSON])
    np.testing.assert_array_equal(keep, [False, True])  # the person was below threshold: the unknown stays


def test_empty_and_degenerate_boxes():
    assert drop_unknown_duplicates(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0, bool)).shape == (0,)
    boxes = np.array([[10, 10, 10, 10], [0, 0, 50, 50]], dtype=float)  # a zero-area unknown box
    keep = drop_unknown_duplicates(boxes, np.array([UNKNOWN, PERSON]), np.ones(2, bool), person_classes=[PERSON])
    assert keep[1]


# --------------------------------------------------------------------------- phantom static objects

from detection_filters import heights_above_floor, line_of_sight_clear, stands_off_floor  # noqa: E402

FY, CY, W, H = 505.5, 240.0, 640, 480


def _scene(cam_h=0.21, box=None):
    """Depth (mm) seen by a level camera ``cam_h`` above a flat floor (background wall at 8 m), with an optional
    ``box = (depth m, height m, u1, u2)`` standing on the floor. A rug is just floor: same depth as under it."""
    v = np.arange(H, dtype=float)[:, None] * np.ones((1, W))
    with np.errstate(divide="ignore"):
        floor = np.where(v > CY, cam_h * FY / (v - CY), np.inf)
    depth = np.minimum(floor, 8.0)
    if box is not None:
        d, bh, u1, u2 = box
        top = CY + (cam_h - bh) * FY / d  # image row of the box's top edge
        bottom = CY + cam_h * FY / d      # where it meets the floor
        depth[int(top):int(bottom), u1:u2] = d
    return (depth * 1000).astype(np.uint16)


def test_a_box_on_the_floor_stands_off_it():
    d = 3.0
    depth = _scene(box=(d, 0.5, 300, 360))
    top = int(CY + (0.21 - 0.5) * FY / d)
    hgt = heights_above_floor(depth, 300, top, 360, int(CY + 0.21 * FY / d) + 5, d, FY, CY, 0.21)
    assert stands_off_floor(hgt) is True
    assert hgt.max() == pytest.approx(0.5, abs=0.02)


def test_a_flat_rug_does_not():
    """A false "unknown" box drawn on the floor 2-3 m ahead: every depth pixel in it is floor."""
    depth = _scene()
    v1, v2 = int(CY + 0.21 * FY / 3.0), int(CY + 0.21 * FY / 2.0)
    hgt = heights_above_floor(depth, 250, v1, 390, v2, 2.5, FY, CY, 0.21)
    assert len(hgt) >= 20
    assert np.abs(hgt).max() < 0.01
    assert stands_off_floor(hgt) is False


def test_the_tall_robot_camera_works_the_same():
    d = 3.0
    depth = _scene(cam_h=0.786, box=(d, 0.6, 300, 360))
    top = int(CY + (0.786 - 0.6) * FY / d)
    hgt = heights_above_floor(depth, 300, top, 360, int(CY + 0.786 * FY / d), d, FY, CY, 0.786)
    assert stands_off_floor(hgt) is True
    depth = _scene(cam_h=0.786)
    v1, v2 = int(CY + 0.786 * FY / 3.0), int(CY + 0.786 * FY / 2.0)
    assert stands_off_floor(heights_above_floor(depth, 250, v1, 390, v2, 2.5, FY, CY, 0.786)) is False


def test_pitch_is_accounted_for():
    """A camera tilted down 5 degrees sees the floor at other depths; with the right pitch it is still flat."""
    pitch = np.radians(5.0)
    v = np.arange(H, dtype=float)[:, None] * np.ones((1, W))
    a = np.arctan((v - CY) / FY) + pitch  # ray angle below horizontal
    with np.errstate(divide="ignore", invalid="ignore"):
        rng = np.where(a > 0, 0.21 / np.sin(a), np.inf)
    z = np.minimum(rng * np.cos(a - pitch), 8.0)  # forward depth along the optical axis
    depth = (z * 1000).astype(np.uint16)
    hgt = heights_above_floor(depth, 250, 330, 390, 400, float(np.median(z[330:400, 250:390])), FY, CY, 0.21, pitch)
    assert np.abs(hgt).max() < 0.02
    assert stands_off_floor(hgt) is False


def test_too_few_pixels_cannot_be_judged():
    assert stands_off_floor(np.zeros(5)) is None
    depth = np.zeros((H, W), np.uint16)
    assert len(heights_above_floor(depth, 0, 0, 50, 50, 2.0, FY, CY, 0.21)) == 0


def test_line_of_sight():
    occ = np.zeros((100, 100), dtype=bool)
    occ[:, 50] = True  # a wall at x in [5.0, 5.1) (0.1 m cells, origin 0, 0)
    assert line_of_sight_clear(occ, 0.1, (0.0, 0.0), (1.0, 5.0), (4.0, 5.0))      # in front of the wall
    assert not line_of_sight_clear(occ, 0.1, (0.0, 0.0), (1.0, 5.0), (7.0, 5.0))  # behind it: a reflection
    assert line_of_sight_clear(occ, 0.1, (0.0, 0.0), (1.0, 5.0), (1.2, 5.0))      # closer than the skip
    assert line_of_sight_clear(occ, 0.1, (0.0, 0.0), (1.0, 5.0), (1.0, 20.0))     # off the map: free
