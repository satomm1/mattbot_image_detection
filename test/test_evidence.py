"""Unit tests for observation_eval/evidence.py: python3 -m pytest mattbot_image_detection/test"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from observation_eval.evidence import (  # noqa: E402
    ABSENT,
    DEPTH_REQUIRE,
    DEPTH_UNINFORMATIVE,
    DEPTH_VETO,
    OUTCOME_ABSENT,
    OUTCOME_INCONCLUSIVE,
    OUTCOME_PRESENT,
    PRESENT,
    UNCERTAIN,
    CameraPose,
    Detection,
    EvalParams,
    FrameEvidence,
    Intrinsics,
    Target,
    bearing,
    classify_depth,
    depth_mode_for,
    frame_verdict,
    in_view,
    match_detections,
    match_known,
    occluded,
    project_roi,
    raycast_background,
    window_outcome,
)

INTR = Intrinsics(fx=505.5, fy=504.4, cx=320.0, cy=240.0, width=640, height=480)  # ~64.7 deg HFOV
CAM = CameraPose(x=0.0, y=0.0, yaw=0.0, height=0.76)  # tall robot, facing +x
CHAIR = Target("chair", 2.0, 0.0, 0.5)
P = EvalParams()


def test_match_class_and_radius():
    dets = [Detection("chair", 2.3, 0.2, 0.5), Detection("chair", 3.0, 0.0, 0.5), Detection("cone", 2.0, 0.0, 0.3)]
    assert match_detections(dets, CHAIR, P) == [(2.3, 0.2, 0.5)]
    assert len(match_detections(dets, CHAIR, EvalParams(match_any_class=True))) == 2


def test_occlusion_only_between_camera_and_object():
    assert occluded([Detection("person", 1.0, 0.1, 0.5)], CAM, CHAIR, P)
    assert not occluded([Detection("person", 1.0, 1.5, 0.5)], CAM, CHAIR, P)  # off to the side
    assert not occluded([Detection("person", 3.0, 0.0, 0.5)], CAM, CHAIR, P)  # behind the object
    assert occluded([Detection("box", 1.0, 0.0, 0.5)], CAM, CHAIR, P)  # any object, not only people


def test_in_view_fov_and_range():
    assert in_view(*bearing(CAM, (2.0, 0.0)), INTR, P)
    assert not in_view(*bearing(CAM, (2.0, 2.0)), INTR, P)  # 45 deg > ~32 deg half FOV
    assert not in_view(*bearing(CAM, (5.0, 0.0)), INTR, P)  # beyond 4.5 m


def test_roi_projection_matches_detector_convention():
    # Detector: x_c = (u - cx) d / fx, theta_object = atan2(x_c, d), bearing = yaw - theta_object.
    # An object to the left (+y) must project to the left of centre (u < cx).
    angle, rng = bearing(CAM, (2.0, 0.5))
    u0, u1, v0, v1 = project_roi(angle, rng, 0.5, CAM.height, INTR, P)
    u_c = (u0 + u1 - 1) / 2.0
    assert u_c < INTR.cx
    d = rng * math.cos(angle)
    theta_object = math.atan2((u_c - INTR.cx) * d / INTR.fx, d)
    assert wrap(CAM.yaw - theta_object) == pytest.approx(angle, abs=0.01)
    # Rows: above the floor, below the horizon for a 0.5 m object seen from 0.76 m
    assert INTR.cy < v0 < v1 <= INTR.height


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def test_roi_excludes_floor_right_behind_object():
    angle, rng = bearing(CAM, (2.0, 0.0))
    _u0, _u1, _v0, v1 = project_roi(angle, rng, 0.5, CAM.height, INTR, P)
    # The lowest ROI row's floor intersection must be >= depth_through_m beyond the object
    z_floor = CAM.height * INTR.fy / ((v1 - 1) - INTR.cy)
    assert z_floor >= 2.0 + P.depth_through_m - 1e-6


def test_depth_modes_by_background_gap():
    assert depth_mode_for(math.inf, P) == DEPTH_REQUIRE
    assert depth_mode_for(0.6, P) == DEPTH_REQUIRE
    assert depth_mode_for(0.4, P) == DEPTH_VETO
    assert depth_mode_for(0.2, P) == DEPTH_UNINFORMATIVE


def depth_image(value):
    return np.full((INTR.height, INTR.width), value, dtype=np.float32)


def test_classify_depth():
    roi = (300, 340, 300, 330)
    assert classify_depth(depth_image(2.0), roi, 2.0, P) == (1.0, 1.0, 0.0, 0.0)  # object there
    assert classify_depth(depth_image(4.0), roi, 2.0, P) == (1.0, 0.0, 1.0, 0.0)  # see-through
    assert classify_depth(depth_image(1.2), roi, 2.0, P) == (1.0, 0.0, 0.0, 1.0)  # something in front
    valid, _, _, _ = classify_depth(depth_image(0.0), roi, 2.0, P)
    assert valid == 0.0


def verdict(dets=(), depth=None, background=math.inf, params=P, loc_ok=True, known=()):
    return frame_verdict(list(dets), CHAIR, CAM, INTR, params, depth, background, loc_ok, known_blockers=known)


def test_frame_verdicts_vision_only():
    vision = EvalParams(use_depth=False)
    assert verdict([Detection("chair", 2.1, 0.0, 0.5)], params=vision).verdict == PRESENT
    assert verdict(params=vision).verdict == ABSENT
    assert verdict([Detection("person", 1.0, 0.0, 0.5)], params=vision).reason == "occluded"
    assert verdict(params=vision, loc_ok=False).reason == "localization"


def test_frame_verdicts_with_depth():
    assert verdict(depth=depth_image(4.0)).verdict == ABSENT  # sees through to 4 m
    assert verdict(depth=depth_image(2.0)).reason == "depth_hit"  # something at the object
    assert verdict(depth=depth_image(0.0)).reason == "depth_invalid"
    assert verdict(depth=None).reason == "depth_missing"
    # Wall 0.4 m behind: veto mode -> wall reading (2.4 m) is neither hit nor required see-through
    assert verdict(depth=depth_image(2.4), background=2.4).verdict == ABSENT
    assert verdict(depth=depth_image(2.0), background=2.4).reason == "depth_hit"
    # Wall right behind (0.2 m): depth uninformative -> vision decides
    ev = verdict(depth=depth_image(2.0), background=2.2)
    assert ev.verdict == ABSENT and ev.depth_mode == DEPTH_UNINFORMATIVE


def frames(n_present=0, n_absent=0, n_uncertain=0):
    return (
        [FrameEvidence(PRESENT, "match", [(0, 0, 0)])] * n_present
        + [FrameEvidence(ABSENT, "clear", [])] * n_absent
        + [FrameEvidence(UNCERTAIN, "occluded", [])] * n_uncertain
    )


def test_window_outcomes():
    assert window_outcome(frames(n_present=2, n_absent=10), P).outcome == OUTCOME_PRESENT
    assert window_outcome(frames(n_absent=8), P).outcome == OUTCOME_ABSENT
    assert window_outcome(frames(n_absent=8, n_uncertain=6), P).outcome == OUTCOME_ABSENT
    out = window_outcome(frames(n_present=1, n_absent=12), P)
    assert out.outcome == OUTCOME_INCONCLUSIVE and "1 matching" in out.reason  # one sighting blocks removal
    assert window_outcome(frames(n_absent=7), P).outcome == OUTCOME_INCONCLUSIVE  # too few frames
    out = window_outcome(frames(n_absent=3, n_uncertain=10), P)
    assert out.outcome == OUTCOME_INCONCLUSIVE and out.uncertain_reasons == {"occluded": 10}


def test_raycast_background():
    blocking = np.zeros((40, 80), dtype=bool)
    blocking[:, 50] = True  # wall at x = 5.0..5.1 (0.1 m cells)
    assert raycast_background(blocking, (0.0, 0.0), 0.1, (1.0, 2.0), 0.0, 6.0) == pytest.approx(4.0, abs=0.06)
    assert raycast_background(blocking, (0.0, 0.0), 0.1, (1.0, 2.0), math.pi / 2, 1.0) == math.inf


def test_new_object_in_front_is_occlusion_not_absence():
    vision = EvalParams(use_depth=False)
    box_in_front = [Detection("box", 1.2, 0.05, 0.4)]
    assert verdict(box_in_front, params=vision).reason == "occluded"
    assert verdict(box_in_front, depth=depth_image(1.2)).reason == "occluded"
    # Detected objects behind the target or off to the side do not occlude it
    assert verdict([Detection("box", 3.0, 0.0, 0.4)], params=vision).verdict == ABSENT
    assert verdict([Detection("box", 1.0, 1.2, 0.4)], params=vision).verdict == ABSENT


def test_undetected_occluder_caught_by_depth_in_every_mode():
    near = depth_image(1.2)  # something 0.8 m in front of the target, detector missed it
    assert verdict(depth=near).reason == "depth_occluded"  # require (open background)
    assert verdict(depth=near, background=2.4).reason == "depth_occluded"  # veto (wall 0.4 m behind)
    ev = verdict(depth=near, background=2.2)  # uninformative (wall 0.2 m behind)
    assert ev.reason == "depth_occluded" and ev.depth_mode == DEPTH_UNINFORMATIVE
    # Known limitation: with depth off, an undetected occluder looks like absence
    assert verdict(params=EvalParams(use_depth=False)).verdict == ABSENT


def test_known_ledger_object_in_front_occludes():
    vision = EvalParams(use_depth=False)
    assert verdict(params=vision, known=[(1.0, 0.0, 0.4)]).reason == "known_occluder"
    assert verdict(params=vision, known=[(1.0, 1.5, 0.4)]).verdict == ABSENT  # not in the way
    assert verdict(params=vision, known=[(3.0, 0.0, 0.4)]).verdict == ABSENT  # behind the target


def test_match_known_groups_by_nearest_object():
    known = {"b": ("box", 1.0, 1.0, 0.4), "c": ("box", 1.6, 1.0, 0.4), "d": ("cone", 3.0, 3.0, 0.3)}
    dets = [Detection("box", 1.1, 1.0, 0.4), Detection("box", 1.5, 1.0, 0.4), Detection("cone", 2.9, 3.0, 0.3),
            Detection("chair", 1.0, 1.0, 0.5)]
    out = match_known(dets, known, P)
    assert out == {"b": [(1.1, 1.0, 0.4)], "c": [(1.5, 1.0, 0.4)], "d": [(2.9, 3.0, 0.3)]}
