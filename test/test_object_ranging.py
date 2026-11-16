"""Unit tests for object_ranging (detector depth / position options): python3 -m pytest mattbot_image_detection/test"""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from object_ranging import (  # noqa: E402
    OBJECT_WINDOW,
    PERSON_WINDOW,
    bbox_percentile_depth,
    box_width_height,
    effective_cx,
    legacy_position,
    pixel_bearing,
    range_in_bearing_span,
    scan_points_map,
    surface_position,
    window_cluster_depth,
)

FX = FY = 505.49
CX, CY = 323.98, 248.60
W, H = 640, 480


# ------------------------------------------------------------------ the original code, copied from
# detect_with_dist_osod.py before the refactor: `legacy` must keep producing exactly this.

def _old_get_depth(x_min, y_min, x_max, y_max, depth):
    depth_values = depth[int(y_min):int(y_max), int(x_min):int(x_max)].flatten()
    depth_values = depth_values[depth_values > 0]
    if len(depth_values) == 0:
        return 0.0
    return np.percentile(depth_values, 25) / 1000.0


def _old_position(trans, theta_camera, x1, y1, x2, y2, estimated_depth, cx, cy, fx, fy):
    x_camera, y_camera = trans[0], trans[1]
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    local_x_min = (x1 - cx) * estimated_depth / fx
    local_z_min = (y1 - cy) * estimated_depth / fy
    local_x_max = (x2 - cx) * estimated_depth / fx
    local_z_max = (y2 - cy) * estimated_depth / fy
    local_x_center = (x_center - cx) * estimated_depth / fx
    local_z_center = (y_center - cy) * estimated_depth / fy  # noqa: F841 (unused in the original too)
    object_width = local_x_max - local_x_min
    object_height = local_z_max - local_z_min
    object_depth = np.min([object_width, object_height, 0.33])
    hypot = np.sqrt(local_x_center**2 + (estimated_depth + object_depth/2)**2)
    theta_object = np.arctan2(local_x_center, estimated_depth)
    x_map = x_camera + (hypot + object_width/2) * np.cos(theta_camera - theta_object)
    y_map = y_camera + (hypot + object_width/2) * np.sin(theta_camera - theta_object)
    return x_map, y_map, object_width, object_height


@pytest.mark.parametrize("seed", range(10))
def test_legacy_is_the_original_code(seed):
    rng = np.random.default_rng(seed)
    depth = rng.integers(0, 6000, (H, W)).astype(np.uint16)
    depth[rng.random((H, W)) < 0.2] = 0
    x1, y1 = int(rng.integers(0, 500)), int(rng.integers(0, 380))
    x2, y2 = x1 + int(rng.integers(10, 140)), y1 + int(rng.integers(10, 100))
    assert bbox_percentile_depth(depth, x1, y1, x2, y2) == _old_get_depth(x1, y1, x2, y2, depth)
    d = float(rng.uniform(0.5, 5.0))
    trans, yaw = (float(rng.uniform(-5, 5)), float(rng.uniform(-5, 5))), float(rng.uniform(-math.pi, math.pi))
    got = legacy_position(trans[0], trans[1], yaw, x1, y1, x2, y2, d, CX, CY, FX, FY)
    want = _old_position(trans, yaw, x1, y1, x2, y2, d, CX, CY, FX, FY)
    assert got == want  # bit-for-bit


# ------------------------------------------------------------------ robust depth


def _scene(person_m=3.0, background_m=6.0, chair_m=None):
    """A person box (x 280-360, y 100-460) at person_m against background_m, floor in its bottom
    rows, speckle zeros; optionally a chair at chair_m covering the box's lower third."""
    rng = np.random.default_rng(0)
    d = np.full((H, W), int(background_m * 1000), np.uint16)
    d[100:460, 280:360] = int(person_m * 1000)
    d[100:460, 280:290] = int(background_m * 1000)  # background beside the arm
    d[300:460, 315:325] = int(background_m * 1000)  # between the legs
    d[440:480, :] = np.linspace(2500, 3200, 40)[:, None].astype(np.uint16)  # the floor
    if chair_m is not None:
        d[340:460, 270:370] = int(chair_m * 1000)
    d[rng.random((H, W)) < 0.05] = 0
    return d


def test_robust_reads_the_person_where_legacy_reads_the_chair():
    d = _scene(person_m=3.0, chair_m=2.0)
    box = (280, 100, 360, 460)
    assert bbox_percentile_depth(d, *box) == pytest.approx(2.0, abs=0.01)  # the occluder in front
    assert window_cluster_depth(d, *box, window=PERSON_WINDOW) == pytest.approx(3.0, abs=0.03)


def test_robust_ignores_floor_gaps_and_speckle():
    d = _scene(person_m=3.0)
    assert window_cluster_depth(d, 280, 100, 360, 460, window=PERSON_WINDOW) == pytest.approx(3.0, abs=0.03)


def test_robust_ignores_a_few_near_pixels():
    d = _scene(person_m=3.0)
    d[150:156, 300:306] = 1000  # a thin thing at 1 m: a few percent of the window
    assert window_cluster_depth(d, 280, 100, 360, 460, window=PERSON_WINDOW) == pytest.approx(3.0, abs=0.03)


def test_robust_object_window_and_no_depth():
    d = np.full((H, W), 4000, np.uint16)
    d[200:300, 200:300] = 1500  # a box-shaped object
    assert window_cluster_depth(d, 200, 200, 300, 300, window=OBJECT_WINDOW) == pytest.approx(1.5, abs=0.03)
    assert window_cluster_depth(np.zeros((H, W), np.uint16), 200, 200, 300, 300) == 0.0


# ------------------------------------------------------------------ positions


def test_surface_position_is_surface_plus_offset_along_the_ray():
    u = CX + 100.0  # right of centre
    x, y, r_surf = surface_position(0.0, 0.0, 0.0, u, 3.0, CX, FX, offset=0.15)
    b = pixel_bearing(u, CX, FX)
    assert b < 0  # right of the optical axis is a negative (clockwise) bearing
    assert r_surf == pytest.approx(3.0 / math.cos(b))
    assert math.hypot(x, y) == pytest.approx(r_surf + 0.15)
    assert math.atan2(y, x) == pytest.approx(b)


def test_legacy_places_a_person_about_04_m_behind_the_surface():
    d = 3.0
    x1, x2 = CX - 0.25 * FX / d, CX + 0.25 * FX / d  # a 0.5 m-wide person straight ahead
    xm, ym, w, _ = legacy_position(0.0, 0.0, 0.0, x1, 100, x2, 460, d, CX, CY, FX, FY)
    assert w == pytest.approx(0.5)
    assert xm - d == pytest.approx(0.33 / 2 + 0.25, abs=1e-6)  # the bias the robust centre removes
    xr, _, _ = surface_position(0.0, 0.0, 0.0, 0.5 * (x1 + x2), d, CX, FX, offset=0.15)
    assert xr - d == pytest.approx(0.15)


def test_bearings_agree_with_the_legacy_direction():
    """surface_position and legacy point the same way (legacy: theta_camera - atan2(local_x, depth))."""
    for u in (50.0, CX, 600.0):
        for yaw in (0.0, 1.0, -2.5):
            xl, yl, _, _ = legacy_position(1.0, 2.0, yaw, u - 1, 100, u + 1, 200, 2.0, CX, CY, FX, FY)
            xr, yr, _ = surface_position(1.0, 2.0, yaw, u, 2.0, CX, FX, offset=0.0)
            assert math.atan2(yl - 2.0, xl - 1.0) == pytest.approx(math.atan2(yr - 2.0, xr - 1.0), abs=1e-9)


def test_rotated_image_uses_the_mirrored_principal_point():
    assert effective_cx(CX, W, rotated=False) == CX
    cxr = effective_cx(CX, W, rotated=True)
    assert cxr == pytest.approx(W - 1 - CX)
    # a column u in the raw image is column (W-1-u) after a 180-degree rotation: same bearing
    for u in (40.0, 320.0, 610.0):
        assert pixel_bearing(W - 1 - u, cxr, FX) == pytest.approx(-pixel_bearing(u, CX, FX))


def test_box_width_height():
    assert box_width_height(100, 100, 200, 300, 2.0, FX, FY) == pytest.approx((100 * 2 / FX, 200 * 2 / FY))


# ------------------------------------------------------------------ LiDAR


def _arc(cx, cy, r, n, a0=0.0, a1=2 * math.pi):
    a = np.linspace(a0, a1, n, endpoint=False)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1)


def _legs(x, y):
    """Front halves of two legs (as a scan sees them), 0.25 m apart."""
    return np.concatenate([_arc(x, y - 0.12, 0.06, 6, math.pi / 2, 3 * math.pi / 2),
                           _arc(x, y + 0.12, 0.06, 6, math.pi / 2, 3 * math.pi / 2)])


def _span(half_width, dist):
    return -math.atan(half_width / dist), math.atan(half_width / dist)


def test_legs_are_ranged():
    pts = np.concatenate([_legs(2.56, 0.0), _arc(0, 0, 6.0, 360)])  # + a wall all around at 6 m
    hit = range_in_bearing_span(pts, 0.0, 0.0, 0.0, *_span(0.3, 2.5))
    assert hit is not None and hit[2] == pytest.approx(2.5, abs=0.06)
    assert abs(hit[1]) < 0.05


def test_the_camera_depth_picks_the_person_behind_a_chair():
    chair = _arc(1.6, 0.0, 0.1, 12, math.pi / 2, 3 * math.pi / 2)
    pts = np.concatenate([chair, _legs(3.06, 0.0)])
    near = range_in_bearing_span(pts, 0.0, 0.0, 0.0, *_span(0.3, 3.0))
    assert near[2] == pytest.approx(1.5, abs=0.06)  # no camera depth: the nearest (safe side)
    person = range_in_bearing_span(pts, 0.0, 0.0, 0.0, *_span(0.3, 3.0), prefer_range=3.0)
    assert person[2] == pytest.approx(3.0, abs=0.06)


def test_nothing_in_the_span_or_too_few_points():
    pts = _legs(2.56, 3.0)  # off to the left, outside the span
    assert range_in_bearing_span(pts, 0.0, 0.0, 0.0, *_span(0.3, 2.5)) is None
    assert range_in_bearing_span(_legs(2.56, 0.0)[:2], 0.0, 0.0, 0.0, *_span(0.3, 2.5)) is None
    assert range_in_bearing_span(np.zeros((0, 2)), 0.0, 0.0, 0.0, -0.1, 0.1) is None


def test_the_margin_keeps_a_neighbour_at_the_box_edge_out():
    lo, hi = _span(0.3, 2.0)  # box edges at +-0.3 m at 2 m
    edge = _arc(1.5, 1.5 * math.tan(hi) * 0.97, 0.03, 6)  # something just inside the left edge, nearer
    pts = np.concatenate([edge, _legs(2.06, 0.0)])
    hit = range_in_bearing_span(pts, 0.0, 0.0, 0.0, lo, hi, margin=0.15)
    assert hit[2] == pytest.approx(2.0, abs=0.06)


def test_a_person_closer_than_the_depth_camera_can_see():
    hit = range_in_bearing_span(_legs(0.46, 0.0), 0.0, 0.0, 0.0, *_span(0.3, 0.4))
    assert hit is not None and hit[2] == pytest.approx(0.4, abs=0.05)


def test_camera_offset_from_the_laser_is_handled():
    """Bearings are taken from the camera: a camera 0.127 m ahead of the laser at a different yaw."""
    pts = _legs(3.0 * math.cos(0.5) + 0.127, 3.0 * math.sin(0.5))
    lo, hi = 0.5 - 0.1, 0.5 + 0.1  # camera at (0.127, 0) facing +x, person at bearing 0.5
    hit = range_in_bearing_span(pts, 0.127, 0.0, 0.0, lo, hi)
    assert hit is not None and hit[2] == pytest.approx(2.95, abs=0.08)


def test_scan_points_map():
    ranges = [1.0, float("inf"), 2.0, 0.0, float("nan")]
    angles = [0.0, 0.1, math.pi / 2, 0.2, 0.3]
    p = scan_points_map(ranges, angles, 1.0, 1.0, math.pi / 2, range_min=0.15, range_max=12.0)
    np.testing.assert_allclose(p, [[1.0, 2.0], [-1.0, 1.0]], atol=1e-12)


def test_robust_spread_depth_falls_back_to_the_densest_surface():
    d = np.zeros((H, W), np.uint16)
    d[200:300, 200:300] = np.linspace(1000, 3000, 100)[None, :].astype(np.uint16)  # a slanted surface, 1-3 m
    d[240:260, 240:260] = 2200  # where most of the window's pixels agree
    got = window_cluster_depth(d, 200, 200, 300, 300, window=OBJECT_WINDOW, min_frac=0.9)
    assert got == pytest.approx(2.2, abs=0.06)  # nothing holds 90 %: the densest surface, not "no depth"
