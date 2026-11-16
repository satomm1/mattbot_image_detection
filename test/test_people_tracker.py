"""Unit tests for people_tracking/tracker.py: python3 -m pytest mattbot_image_detection/test"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from people_tracking.tracker import PeopleTracker, TrackerParams  # noqa: E402

HZ = 5.0  # the detector's rate on the robot, roughly


def _walk(tracker, seconds, pos_fn, noise=0.1, seed=0, t0=0.0, width=0.5):
    rng = np.random.default_rng(seed)
    n = int(round(seconds * HZ))
    for k in range(n + 1):
        t = t0 + k / HZ
        pts = [np.asarray(p, dtype=float) + rng.normal(0.0, noise, 2) for p in pos_fn(t)]
        tracker.update(t, pts, [width] * len(pts))
    return t0 + n / HZ


def test_a_walker_gets_its_velocity():
    tr = PeopleTracker()
    t = _walk(tr, 3.0, lambda t: [(1.0 + 0.8 * t, 2.0)], noise=0.1)
    (p,) = tr.confirmed()
    assert p.x[2] == pytest.approx(0.8, abs=0.15) and p.x[3] == pytest.approx(0.0, abs=0.15)
    assert math.hypot(p.x[0] - (1.0 + 0.8 * t), p.x[1] - 2.0) < 0.1  # below the measurement noise
    assert p.vel_std < 0.35  # the steady-state filter uncertainty at 5 Hz, accel_std 1, meas_std 0.15


def test_velocity_converges_within_about_a_second():
    errs = []
    for seed in range(10):
        tr = PeopleTracker()
        _walk(tr, 1.2, lambda t: [(0.6 * t, -0.6 * t)], noise=0.1, seed=seed)
        (p,) = tr.confirmed()
        errs.append(math.hypot(p.x[2] - 0.6, p.x[3] + 0.6))
    assert np.median(errs) < 0.25


def test_standing_still_reads_zero_velocity():
    tr = PeopleTracker()
    _walk(tr, 3.0, lambda t: [(3.0, 1.0)], noise=0.1)
    (p,) = tr.confirmed()
    assert math.hypot(p.x[2], p.x[3]) < 0.15


def test_stop_and_go_slows_to_zero():
    tr = PeopleTracker()
    t = _walk(tr, 2.0, lambda t: [(0.7 * t, 0.0)], noise=0.05)
    _walk(tr, 2.0, lambda t: [(0.7 * 2.0, 0.0)], noise=0.05, t0=t + 1 / HZ, seed=1)
    (p,) = tr.confirmed()
    assert math.hypot(p.x[2], p.x[3]) < 0.2


def test_two_people_crossing_keep_their_ids():
    tr = PeopleTracker()

    def two(t):
        return [(-2.0 + 0.8 * t, 0.6), (2.0 - 0.8 * t, -0.6)]  # pass each other 1.2 m apart

    _walk(tr, 1.0, two, noise=0.05)
    ids = {bool(p.x[2] > 0): p.id for p in tr.confirmed()}  # by direction of travel
    assert len(ids) == 2
    _walk(tr, 4.0, two, noise=0.05, t0=1.0 + 1 / HZ, seed=3)
    now = {bool(p.x[2] > 0): p.id for p in tr.confirmed()}
    assert now == ids


def test_missed_frames_coast_then_expire():
    tr = PeopleTracker(TrackerParams(max_coast_s=1.0))
    t = _walk(tr, 2.0, lambda t: [(1.0 * t, 0.0)], noise=0.02)
    (p,) = tr.confirmed()
    # 0.6 s of empty frames: still there, predicted forward
    for k in range(1, 4):
        tr.update(t + k / HZ, [])
    (q,) = tr.confirmed()
    assert q.id == p.id and q.x[0] > p.x[0] + 0.4 and q.last_seen == pytest.approx(t)
    tr.update(t + 1.2, [])
    assert tr.confirmed() == []


def test_confirmed_predicts_without_advancing_the_filter():
    tr = PeopleTracker()
    t = _walk(tr, 2.0, lambda t: [(1.0 * t, 0.0)], noise=0.02)
    (now,) = tr.confirmed()
    (later,) = tr.confirmed(t + 0.5)
    assert later.x[0] == pytest.approx(now.x[0] + 0.5 * now.x[2])
    assert tr.confirmed()[0].x[0] == now.x[0]


def test_a_single_false_detection_never_confirms():
    tr = PeopleTracker(TrackerParams(min_hits=2))
    tr.update(0.0, [(5.0, 5.0)])
    tr.update(0.2, [])
    tr.update(0.4, [])
    assert tr.confirmed() == [] and tr.tracks == []


def test_speed_is_clamped():
    tr = PeopleTracker(TrackerParams(max_speed=2.0, gate_m=3.0, gate_chi2=1e6))
    _walk(tr, 1.0, lambda t: [(4.0 * t, 0.0)], noise=0.0)  # a 4 m/s "person" (e.g. bad association)
    (p,) = tr.confirmed()
    assert math.hypot(p.x[2], p.x[3]) <= 2.0 + 1e-9


def test_far_detections_start_new_tracks_and_bad_input_is_ignored():
    tr = PeopleTracker(TrackerParams(gate_m=1.0))
    tr.update(0.0, [(0.0, 0.0)])
    tr.update(0.2, [(3.0, 0.0), (float("nan"), 1.0)])  # a jump beyond the gate; a NaN detection
    assert len(tr.tracks) == 1  # the tentative first track expired at its miss; the jump is a new one
    assert tr.tracks[0].x[0] == 3.0
    tr.update(0.1, [(0.0, 0.0)])  # an out-of-order (older) frame is dropped
    assert tr.t == 0.2


def test_radius_follows_the_detection_width():
    tr = PeopleTracker(TrackerParams(width_alpha=1.0))
    _walk(tr, 1.0, lambda t: [(0.0, 0.0)], noise=0.0, width=0.6)
    assert tr.confirmed()[0].radius == pytest.approx(0.3)


def test_bad_params_are_refused():
    with pytest.raises(ValueError):
        TrackerParams(meas_std=0.0)
    with pytest.raises(ValueError):
        TrackerParams(min_hits=0)
