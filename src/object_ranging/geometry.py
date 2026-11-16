"""From a box and a depth to a map-frame position (camera_link is body-aligned: +x forward)."""

import math

import numpy as np


def effective_cx(cx, image_width, rotated):
    """The principal point in the image the boxes are in: mirrored when the image was rotated
    180 degrees (the tall robot's upside-down camera)."""
    return (image_width - 1) - cx if rotated else cx


def pixel_bearing(u, cx, fx):
    """Bearing of image column ``u`` from the optical axis, rad, positive to the camera's LEFT
    (CCW, as map yaw): a column right of the principal point is a negative bearing."""
    return -math.atan((u - cx) / fx)


def box_width_height(x1, y1, x2, y2, depth, fx, fy):
    """The box's metric width and height at ``depth`` (as the original code computes them)."""
    return (x2 - x1) * depth / fx, (y2 - y1) * depth / fy


def legacy_position(x_camera, y_camera, theta_camera, x1, y1, x2, y2, estimated_depth, cx, cy, fx, fy):
    """LEGACY (the original in-script maths, verbatim). Returns ``(x_map, y_map, width, height)``.

    The centre lands ``min(w, h, 0.33)/2 + w/2`` past the surface: about 0.4 m for a 0.5 m-wide person.
    """
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    local_x_min = (x1 - cx) * estimated_depth / fx  # meters (left-most point)
    local_z_min = (y1 - cy) * estimated_depth / fy  # meters (bottom-most point)
    local_x_max = (x2 - cx) * estimated_depth / fx  # meters (right-most point)
    local_z_max = (y2 - cy) * estimated_depth / fy  # meters (top-most point)
    local_x_center = (x_center - cx) * estimated_depth / fx  # meters
    object_width = local_x_max - local_x_min
    object_height = local_z_max - local_z_min
    object_depth = np.min([object_width, object_height, 0.33])
    hypot = np.sqrt(local_x_center**2 + (estimated_depth + object_depth/2)**2)
    theta_object = np.arctan2(local_x_center, estimated_depth)
    x_map = x_camera + (hypot + object_width/2) * np.cos(theta_camera - theta_object)
    y_map = y_camera + (hypot + object_width/2) * np.sin(theta_camera - theta_object)
    return x_map, y_map, object_width, object_height


def ray_point(x_camera, y_camera, theta_camera, bearing, rng):
    """The map point at range ``rng`` along ``bearing`` (camera-relative, CCW) from the camera."""
    a = theta_camera + bearing
    return x_camera + rng * math.cos(a), y_camera + rng * math.sin(a)


def surface_position(x_camera, y_camera, theta_camera, u_center, depth, cx, fx, offset):
    """ROBUST: the centre ``offset`` m behind the surface at forward ``depth``, along the ray
    through column ``u_center``. Returns ``(x_map, y_map, range_to_surface)``."""
    bearing = pixel_bearing(u_center, cx, fx)
    surface_range = depth / math.cos(bearing)  # depth is along the optical axis; range is along the ray
    x, y = ray_point(x_camera, y_camera, theta_camera, bearing, surface_range + offset)
    return x, y, surface_range
