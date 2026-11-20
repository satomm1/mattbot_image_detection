"""Unit tests for detection_filters (per-class thresholds, open-set duplicate removal)."""

import os
import sys

import numpy as np

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
