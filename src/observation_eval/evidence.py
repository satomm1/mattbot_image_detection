"""Is a known object still there? Per-frame evidence and per-window decision.

Pure numpy (no ROS) so it can be unit-tested. Used by scripts/observation_evaluator.py during
the navigator's observation windows (robot stopped, facing the object).

Each detector frame gets a verdict:
  PRESENT    a detection of the object's class near the object
  ABSENT     no such detection, the object is in view, nothing occludes it, and the depth
             check (if enabled) agrees
  UNCERTAIN  occluded, out of view / range, localization lost, or depth inconclusive
Occluders: any detection (not only people) or known ledger object standing between the camera
and the object, or depth pixels reading clearly nearer than the object. A hidden object is
"occluded", never "gone".
A window is PRESENT / ABSENT / INCONCLUSIVE from the counts of frame verdicts.

Depth check (optional). The image region where the object should appear is compared with the
expected depths. Only rows where empty space would read clearly deeper than the object are used
(the floor right behind the object would otherwise look like the object). How depth is used per
frame depends on how far the background (static-map wall) is behind the object:
  gap >= depth_through_m    "require": ABSENT needs most pixels to see through the object's spot
  depth_tol_m < gap         "veto": ABSENT is blocked only if pixels show something at the object
  gap <= depth_tol_m        "uninformative": wall indistinguishable from the object; vision
                            only (but pixels nearer than the object still mean "occluded")
Without depth, an occluder the detector misses cannot be told apart from the object being gone.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

# Frame verdicts
PRESENT = 0
ABSENT = 1
UNCERTAIN = 2

# Window outcomes (same values as mattbot_dds/ObservationResult)
OUTCOME_PRESENT = 0
OUTCOME_ABSENT = 1
OUTCOME_INCONCLUSIVE = 2

# Depth modes per frame
DEPTH_OFF = "off"
DEPTH_REQUIRE = "require"
DEPTH_VETO = "veto"
DEPTH_UNINFORMATIVE = "uninformative"


@dataclass
class EvalParams:
    # Matching
    match_radius_m: float = 0.75  # same as the ledger's match radius
    match_any_class: bool = False
    # Visibility
    max_range_m: float = 4.5  # detector reports objects up to 5 m
    fov_margin_rad: float = 0.05
    occlusion_margin_m: float = 0.3  # a person this much closer than the object can occlude it
    person_width_m: float = 0.5
    # Depth
    use_depth: bool = True
    object_height_m: float = 0.5
    min_object_height_m: float = 0.05  # ignore pixels this close to the floor
    depth_tol_m: float = 0.3  # |depth - expected| below this = "hit" (something at the object)
    depth_through_m: float = 0.5  # deeper than expected by this = "see-through"
    depth_through_fraction: float = 0.7
    depth_hit_fraction: float = 0.3
    depth_occluded_fraction: float = 0.3  # pixels nearer than the object: something in front
    depth_min_valid: float = 0.3  # fraction of ROI pixels with a depth reading
    roi_width_shrink: float = 0.8  # use the central part of the object's width
    # Window decision
    min_present_frames: int = 2
    min_absent_frames: int = 8
    absent_fraction: float = 0.8  # of frames that were not UNCERTAIN


@dataclass
class Target:
    class_name: str
    x: float  # map frame
    y: float
    width: float


@dataclass
class CameraPose:
    x: float  # map frame
    y: float
    yaw: float
    height: float  # camera height above the floor


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def hfov(self):
        return 2.0 * math.atan(self.width / (2.0 * self.fx))


@dataclass
class Detection:
    class_name: str
    x: float  # map frame
    y: float
    width: float


@dataclass
class FrameEvidence:
    verdict: int
    reason: str  # "match", "clear", or why UNCERTAIN
    matches: List[Tuple[float, float, float]]  # (x, y, width) of matching detections
    depth_mode: str = DEPTH_OFF
    depth_valid: float = 0.0
    depth_hit: float = 0.0
    depth_through: float = 0.0
    depth_near: float = 0.0


@dataclass
class WindowOutcome:
    outcome: int
    reason: str
    frames: int
    present: int
    absent: int
    uncertain: int
    uncertain_reasons: dict


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def bearing(cam, target_xy):
    """(angle relative to the optical axis, range) of a map point."""
    dx, dy = target_xy[0] - cam.x, target_xy[1] - cam.y
    return wrap_angle(math.atan2(dy, dx) - cam.yaw), math.hypot(dx, dy)


def match_detections(detections, target, params):
    """Detections of the target's class within match_radius_m of it."""
    out = []
    for d in detections:
        if not params.match_any_class and d.class_name != target.class_name:
            continue
        if math.hypot(d.x - target.x, d.y - target.y) <= params.match_radius_m:
            out.append((d.x, d.y, d.width))
    return out


def blocks_sight(cam, target, x, y, width, params):
    """Does something at (x, y) of this width stand between the camera and the target?"""
    tx, ty = target.x - cam.x, target.y - cam.y
    dist = math.hypot(tx, ty)
    if dist <= 0:
        return False
    ux, uy = tx / dist, ty / dist
    px, py = x - cam.x, y - cam.y
    along = px * ux + py * uy
    lateral = abs(-px * uy + py * ux)
    return 0.0 < along < dist - params.occlusion_margin_m and lateral < (target.width + width) / 2.0


def occluded(detections, cam, target, params):
    """Any detection (person or object) stands between the camera and the object."""
    for d in detections:
        width = max(d.width, params.person_width_m) if d.class_name == "person" else d.width
        if blocks_sight(cam, target, d.x, d.y, width, params):
            return True
    return False


def known_occluded(known_blockers, cam, target, params):
    """A known ledger object [(x, y, width), ...] stands between the camera and the object."""
    return any(blocks_sight(cam, target, x, y, w, params) for x, y, w in known_blockers)


def match_known(detections, known_objects, params):
    """Detections matching known ledger objects {object_id: (class_name, x, y, width)}.

    Returns {object_id: [(x, y, width), ...]}; each detection counts for its nearest match.
    """
    out = {}
    for d in detections:
        best, best_d = None, params.match_radius_m
        for object_id, (class_name, x, y, _w) in known_objects.items():
            if not params.match_any_class and d.class_name != class_name:
                continue
            dist = math.hypot(d.x - x, d.y - y)
            if dist <= best_d:
                best, best_d = object_id, dist
        if best is not None:
            out.setdefault(best, []).append((d.x, d.y, d.width))
    return out


def in_view(angle, rng, intr, params):
    return rng <= params.max_range_m and abs(angle) <= intr.hfov / 2.0 - params.fov_margin_rad


def project_roi(angle, rng, target_width, cam_height, intr, params):
    """Pixel box (u0, u1, v0, v1), exclusive ends, where empty space would read clearly deeper.

    Columns: the central roi_width_shrink of the object's width. Rows: from the object's top
    down to where the floor behind the object is still >= depth_through_m beyond it (and at
    least min_object_height_m above the floor). Returns None if empty.
    """
    z = rng * math.cos(angle)  # depth along the optical axis
    if z <= 0:
        return None
    u_c = intr.cx - intr.fx * math.tan(angle)
    half = intr.fx * (target_width * params.roi_width_shrink / 2.0) / z
    u0, u1 = int(math.floor(u_c - half)), int(math.ceil(u_c + half)) + 1
    v_top = intr.cy + intr.fy * (cam_height - params.object_height_m) / z
    v_floor_far = intr.cy + intr.fy * cam_height / (z + params.depth_through_m)
    v_low = intr.cy + intr.fy * (cam_height - params.min_object_height_m) / z
    v0 = int(math.ceil(v_top))
    v1 = int(math.floor(min(v_floor_far, v_low))) + 1
    u0, u1 = max(u0, 0), min(u1, intr.width)
    v0, v1 = max(v0, 0), min(v1, intr.height)
    if u0 >= u1 or v0 >= v1:
        return None
    return u0, u1, v0, v1


def depth_mode_for(gap_m, params):
    """How depth can be used given the background distance behind the object."""
    if gap_m >= params.depth_through_m:
        return DEPTH_REQUIRE
    if gap_m > params.depth_tol_m:
        return DEPTH_VETO
    return DEPTH_UNINFORMATIVE


def classify_depth(depth_m, roi, expected_z, params):
    """(valid, hit, through, near) fractions of the depth image (metres, 0 = no reading) in roi.

    near: pixels clearly nearer than the object, i.e. something standing in front of it (the ROI
    rows are chosen so the floor cannot read nearer than the object).
    """
    u0, u1, v0, v1 = roi
    patch = np.asarray(depth_m[v0:v1, u0:u1], dtype=np.float64)
    valid = np.isfinite(patch) & (patch > 0)
    n_valid = int(valid.sum())
    valid_frac = n_valid / patch.size if patch.size else 0.0
    if n_valid == 0:
        return valid_frac, 0.0, 0.0, 0.0
    d = patch[valid]
    hit = float(np.mean(np.abs(d - expected_z) < params.depth_tol_m))
    through = float(np.mean(d > expected_z + params.depth_through_m))
    near = float(np.mean(d < expected_z - params.depth_tol_m))
    return valid_frac, hit, through, near


def frame_verdict(
    detections: List[Detection],
    target: Target,
    cam: CameraPose,
    intr: Intrinsics,
    params: EvalParams,
    depth_m: Optional[np.ndarray] = None,
    background_m: float = math.inf,
    localization_ok: bool = True,
    known_blockers=(),
):
    """Evidence from one detector frame. background_m: static-map distance from the camera to the
    first wall along the bearing to the object (inf if none within range). known_blockers: other
    active ledger objects [(x, y, width), ...]."""
    matches = match_detections(detections, target, params)
    if matches:
        return FrameEvidence(PRESENT, "match", matches)
    if not localization_ok:
        return FrameEvidence(UNCERTAIN, "localization", [])
    angle, rng = bearing(cam, (target.x, target.y))
    if not in_view(angle, rng, intr, params):
        return FrameEvidence(UNCERTAIN, "out_of_view", [])
    if occluded(detections, cam, target, params):
        return FrameEvidence(UNCERTAIN, "occluded", [])
    if known_occluded(known_blockers, cam, target, params):
        return FrameEvidence(UNCERTAIN, "known_occluder", [])
    if not params.use_depth:
        return FrameEvidence(ABSENT, "clear", [])

    mode = depth_mode_for(background_m - rng, params)
    roi = project_roi(angle, rng, target.width, cam.height, intr, params) if depth_m is not None else None
    if mode == DEPTH_UNINFORMATIVE:
        # The wall behind is indistinguishable from the object, but something in front is not
        if roi is not None:
            valid, hit, through, near = classify_depth(depth_m, roi, rng * math.cos(angle), params)
            ev = dict(depth_mode=mode, depth_valid=valid, depth_hit=hit, depth_through=through, depth_near=near)
            if valid >= params.depth_min_valid and near >= params.depth_occluded_fraction:
                return FrameEvidence(UNCERTAIN, "depth_occluded", [], **ev)
            return FrameEvidence(ABSENT, "clear", [], **ev)
        return FrameEvidence(ABSENT, "clear", [], depth_mode=mode)
    if depth_m is None:
        return FrameEvidence(UNCERTAIN, "depth_missing", [], depth_mode=mode)
    if roi is None:
        return FrameEvidence(UNCERTAIN, "depth_no_roi", [], depth_mode=mode)
    z = rng * math.cos(angle)
    valid, hit, through, near = classify_depth(depth_m, roi, z, params)
    ev = dict(depth_mode=mode, depth_valid=valid, depth_hit=hit, depth_through=through, depth_near=near)
    if valid < params.depth_min_valid:
        return FrameEvidence(UNCERTAIN, "depth_invalid", [], **ev)
    if near >= params.depth_occluded_fraction:
        return FrameEvidence(UNCERTAIN, "depth_occluded", [], **ev)
    if hit >= params.depth_hit_fraction:
        return FrameEvidence(UNCERTAIN, "depth_hit", [], **ev)
    if mode == DEPTH_REQUIRE and through < params.depth_through_fraction:
        return FrameEvidence(UNCERTAIN, "depth_not_through", [], **ev)
    return FrameEvidence(ABSENT, "clear", [], **ev)


def window_outcome(frames: List[FrameEvidence], params: EvalParams):
    """Decide PRESENT / ABSENT / INCONCLUSIVE from a window's frame evidence."""
    present = sum(f.verdict == PRESENT for f in frames)
    absent = sum(f.verdict == ABSENT for f in frames)
    uncertain = len(frames) - present - absent
    reasons = {}
    for f in frames:
        if f.verdict == UNCERTAIN:
            reasons[f.reason] = reasons.get(f.reason, 0) + 1
    out = dict(frames=len(frames), present=present, absent=absent, uncertain=uncertain, uncertain_reasons=reasons)

    if present >= params.min_present_frames:
        return WindowOutcome(OUTCOME_PRESENT, "%d matching frames" % present, **out)
    decided = present + absent
    if present == 0 and absent >= params.min_absent_frames and absent >= params.absent_fraction * decided:
        return WindowOutcome(OUTCOME_ABSENT, "%d/%d clear frames" % (absent, len(frames)), **out)
    if present > 0:
        reason = "only %d matching frame(s)" % present
    elif absent < params.min_absent_frames:
        reason = "only %d clear frame(s) (need %d)" % (absent, params.min_absent_frames)
    else:
        reason = "clear frames %d/%d below fraction" % (absent, decided)
    return WindowOutcome(OUTCOME_INCONCLUSIVE, reason, **out)


def raycast_background(blocking, origin, resolution, start_xy, angle_map, max_range_m):
    """Distance from start_xy along map angle angle_map to the first blocking cell (inf if none)."""
    height, width = blocking.shape
    step = resolution / 2.0
    d = np.arange(step, max_range_m + step, step)
    xs = start_xy[0] + math.cos(angle_map) * d
    ys = start_xy[1] + math.sin(angle_map) * d
    ci = np.floor((xs - origin[0]) / resolution).astype(int)
    cj = np.floor((ys - origin[1]) / resolution).astype(int)
    on_map = (ci >= 0) & (ci < width) & (cj >= 0) & (cj < height)
    hit = ~on_map
    hit[on_map] = blocking[cj[on_map], ci[on_map]]
    idx = np.flatnonzero(hit)
    return float(d[idx[0]]) if len(idx) else math.inf
