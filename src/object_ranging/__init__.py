"""Where a detected object is: depth / position estimators for detect_with_dist_osod.py (ROS-free).

Selectable per run (detector params ``~range_method`` and ``~person_range_source``):

* ``legacy``  -- the original estimate, kept verbatim: the 25th percentile of every non-zero
  depth pixel in the box, centre ``hypot(lateral, d + min(w, h, 0.33)/2) + w/2`` along the ray.
* ``robust``  -- depth of the nearest substantial surface in a central window of the box (a
  torso window for people), centre one body half-depth behind that surface.
* LiDAR ranging for people (``person_range_source: lidar``) -- the nearest ``/scan`` cluster
  inside the box's bearing span (consistent with the camera depth when there is one), with the
  camera estimate as the fallback. Works below the depth camera's minimum range and past 5 m.
"""

from .depth import OBJECT_WINDOW, PERSON_WINDOW, bbox_percentile_depth, window_cluster_depth
from .geometry import (box_width_height, effective_cx, legacy_position, pixel_bearing, ray_point,
                       surface_position)
from .lidar import range_in_bearing_span, scan_points_map

__all__ = ["OBJECT_WINDOW", "PERSON_WINDOW", "bbox_percentile_depth", "box_width_height", "effective_cx",
           "legacy_position", "pixel_bearing", "range_in_bearing_span", "ray_point", "scan_points_map",
           "surface_position", "window_cluster_depth"]
