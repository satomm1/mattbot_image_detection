"""Unit tests for detection_timing (the detector's turning gate and pairing stats)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from detection_timing import PairStats, TurningGate  # noqa: E402


def _drive(gate, t0, t1, w, hz=50.0):
    n = int(round((t1 - t0) * hz))
    for k in range(n + 1):
        gate.add_odom(t0 + k / hz, w)


def test_steering_corrections_do_not_skip_frames():
    g = TurningGate(threshold=0.8)
    _drive(g, 0.0, 2.0, 0.5)  # the old gate (0.10 rad/s commanded) skipped all of these
    assert not g.turning(1.9)


def test_a_fast_turn_skips_frames_and_its_cooldown():
    g = TurningGate(threshold=0.8, cooldown=0.15)
    _drive(g, 0.0, 1.0, 0.0)
    _drive(g, 1.02, 1.5, 1.5)
    _drive(g, 1.52, 3.0, 0.0)
    assert not g.turning(0.9)
    assert g.turning(1.3)
    assert g.turning(1.6)        # within the 0.15 s cooldown after the turn
    assert not g.turning(1.8)


def test_the_gate_is_evaluated_at_the_image_time_not_now():
    g = TurningGate(threshold=0.8, cooldown=0.1)
    _drive(g, 0.0, 1.0, 1.5)     # turning until t = 1.0
    _drive(g, 1.02, 2.0, 0.0)    # then straight
    assert g.turning(0.95)       # a frame taken during the turn, processed late, is still skipped
    assert not g.turning(1.5)


def test_without_odometry_frames_are_never_skipped():
    g = TurningGate()
    assert not g.turning(5.0)
    _drive(g, 0.0, 1.0, 2.0)
    assert not g.turning(10.0)   # odometry stopped long ago: no information, do not skip


def test_negative_yaw_rates_count():
    g = TurningGate(threshold=0.8)
    _drive(g, 0.0, 1.0, -1.2)
    assert g.turning(0.9)


def test_out_of_order_odometry_is_ignored():
    g = TurningGate(threshold=0.8)
    g.add_odom(1.0, 0.0)
    g.add_odom(0.5, 3.0)
    assert not g.turning(0.55)


def test_bad_params_are_refused():
    with pytest.raises(ValueError):
        TurningGate(threshold=0.0)


def test_pair_stats():
    s = PairStats()
    assert s.summary() is None
    s.add_pair(10.000, 10.012)
    s.add_pair(10.033, 10.030)
    s.add_pair(10.066, 10.086)
    med, mx = s.summary()
    assert med == pytest.approx(0.012) and mx == pytest.approx(0.020) and s.n_pairs == 3
