#!/usr/bin/env python3
"""Decides whether a ledger object is still there during the navigator's observation windows.

Inputs:  /observation/events    STARTED / ENDED / ABORTED windows (localize_and_navigate.py)
         /detected_objects      detector output; each message is one frame
         /object_beliefs        class, position and width of each ledger object
         /camera/depth/image_raw, /camera/color/camera_info   (only with ~use_depth)
         /map                   static map, for the background distance behind the object
         /lost_localization     frames after this are UNCERTAIN
         TF map -> camera_link
Outputs: /confirmed_objects     re-sighting (PRESENT): the ledger resets the object's belief;
                                also for other known objects seen during the window
         /removed_objects       removal (ABSENT): the ledger removes the object on all robots
         /observation/results   mattbot_dds/ObservationResult for every window

The per-frame and per-window logic is in observation_eval/evidence.py.
"""

import math
import threading
import time
from dataclasses import fields

import numpy as np
import rospy
import tf
from geometry_msgs.msg import Pose
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool
from mattbot_dds.msg import ObjectBeliefArray, ObservationEvent, ObservationResult
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray

from observation_eval.evidence import (
    OUTCOME_ABSENT,
    OUTCOME_PRESENT,
    CameraPose,
    Detection,
    EvalParams,
    Intrinsics,
    Target,
    bearing,
    frame_verdict,
    match_known,
    raycast_background,
    window_outcome,
)

LOST_LOCALIZATION_HOLD_S = 2.0  # frames this soon after /lost_localization are UNCERTAIN


class ObservationEvaluatorNode:
    def __init__(self):
        rospy.init_node("observation_evaluator")
        defaults = EvalParams()
        self.params = EvalParams(**{
            f.name: type(getattr(defaults, f.name))(rospy.get_param("~" + f.name, getattr(defaults, f.name)))
            for f in fields(EvalParams)
        })
        self.tall = bool(rospy.get_param("~tall", False))  # images rotated 180 deg, as in the detector
        self.depth_max_age_s = float(rospy.get_param("~depth_max_age_s", 0.5))
        self.resight_min_interval_s = float(rospy.get_param("~resight_min_interval_s", 30.0))
        self.last_resight = {}  # object_id -> wall time of the last re-sighting we sent

        self.lock = threading.Lock()
        self.windows = {}  # object_id -> {"target", "start", "frames", "params", "others"}
        self.objects = {}  # object_id -> (class_name, x, y, width) from /object_beliefs (local frame)
        self.intrinsics = None
        self.depth = None  # (wall time, HxW metres)
        self.blocking = None
        self.map_info = None
        self.last_lost = -math.inf

        self.tf_listener = tf.TransformListener()
        self.confirm_pub = rospy.Publisher("/confirmed_objects", DetectedObject, queue_size=10)
        self.remove_pub = rospy.Publisher("/removed_objects", DetectedObject, queue_size=10)
        self.result_pub = rospy.Publisher("/observation/results", ObservationResult, queue_size=10)

        rospy.Subscriber("/observation/events", ObservationEvent, self.event_callback, queue_size=20)
        rospy.Subscriber("/detected_objects", DetectedObjectArray, self.detections_callback, queue_size=5)
        rospy.Subscriber("/object_beliefs", ObjectBeliefArray, self.beliefs_callback, queue_size=1)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback, queue_size=1)
        rospy.Subscriber("/map", OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber("/lost_localization", Bool, self.lost_callback, queue_size=5)
        if self.params.use_depth:
            rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_callback, queue_size=1, buff_size=2 ** 24)
        rospy.loginfo("observation_evaluator: depth check %s", "on" if self.params.use_depth else "off")

    # ---------- Inputs ----------

    def beliefs_callback(self, msg):
        objects = {o.object_id: (o.class_name, o.local_x, o.local_y, float(o.width)) for o in msg.objects}
        with self.lock:
            self.objects = objects

    def camera_info_callback(self, msg):
        if self.intrinsics is not None:
            return
        fx, fy, cx, cy = msg.K[0], msg.K[4], msg.K[2], msg.K[5]
        if self.tall:  # image rotated 180 deg: principal point mirrors
            cx, cy = msg.width - 1 - cx, msg.height - 1 - cy
        self.intrinsics = Intrinsics(fx, fy, cx, cy, msg.width, msg.height)

    def map_callback(self, msg):
        grid = np.asarray(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        with self.lock:
            self.blocking = (grid >= 50) | (grid < 0)
            self.map_info = msg.info

    def lost_callback(self, msg):
        if msg.data:
            self.last_lost = time.time()

    def depth_callback(self, msg):
        with self.lock:
            if not self.windows:
                return  # only needed while observing
        if msg.encoding == "16UC1":
            depth = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width).astype(np.float32) / 1000.0
        elif msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.width)
        else:
            rospy.logwarn_throttle(60, "observation_evaluator: unsupported depth encoding %s", msg.encoding)
            return
        if self.tall:
            depth = depth[::-1, ::-1]
        with self.lock:
            self.depth = (time.time(), depth)

    # ---------- Windows ----------

    def event_callback(self, msg):
        with self.lock:
            if msg.event == ObservationEvent.STARTED:
                class_name, _x, _y, width = self.objects.get(msg.object_id, ("", 0.0, 0.0, 0.5))
                params = self.params
                if not class_name:  # unknown class: accept any detected class near the object
                    params = EvalParams(**{**params.__dict__, "match_any_class": True})
                self.windows[msg.object_id] = {
                    "target": Target(class_name, msg.target_x, msg.target_y, width),
                    "start": msg.window_start or time.time(),
                    "frames": [],
                    "params": params,
                    "others": {},  # other known object_id -> [(x, y, width) per matching detection, frame count]
                }
                return
            window = self.windows.pop(msg.object_id, None)
        if window is None or msg.event != ObservationEvent.ENDED:
            return  # ABORTED, or a window we did not see start
        self.decide(msg.object_id, window, msg.window_end or time.time())

    def camera_pose(self):
        try:
            (trans, rot) = self.tf_listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None
        yaw = tf.transformations.euler_from_quaternion(rot)[2]
        return CameraPose(trans[0], trans[1], yaw, trans[2])

    def detections_callback(self, msg):
        with self.lock:
            if not self.windows or self.intrinsics is None:
                return
        cam = self.camera_pose()
        if cam is None:
            return
        dets = [Detection(o.class_name, o.pose.position.x, o.pose.position.y, o.width) for o in msg.objects]
        now = time.time()
        with self.lock:
            depth = None
            if self.depth is not None and now - self.depth[0] <= self.depth_max_age_s:
                depth = self.depth[1]
            intr = self.depth_intrinsics(depth)
            for target_id, window in self.windows.items():
                target, params = window["target"], window["params"]
                others = {oid: o for oid, o in self.objects.items() if oid != target_id}
                ev = frame_verdict(
                    dets, target, cam, intr, params,
                    depth_m=depth,
                    background_m=self.background(cam, target, params),
                    localization_ok=now - self.last_lost > LOST_LOCALIZATION_HOLD_S
                    and self.last_lost < window["start"],
                    known_blockers=[(x, y, w) for _c, x, y, w in others.values()],
                )
                window["frames"].append(ev)
                # Other known objects seen in this frame (positive evidence only)
                for oid, matches in match_known(dets, others, params).items():
                    entry = window["others"].setdefault(oid, [[], 0])
                    entry[0].extend(matches)
                    entry[1] += 1

    def depth_intrinsics(self, depth):
        """Camera intrinsics scaled to the depth image size (if it differs from the colour image)."""
        intr = self.intrinsics
        if depth is None or (depth.shape[1] == intr.width and depth.shape[0] == intr.height):
            return intr
        sx, sy = depth.shape[1] / intr.width, depth.shape[0] / intr.height
        return Intrinsics(intr.fx * sx, intr.fy * sy, intr.cx * sx, intr.cy * sy, depth.shape[1], depth.shape[0])

    def background(self, cam, target, params):
        """Static-map distance from the camera to the first wall along the bearing to the object."""
        if self.blocking is None:
            return math.inf
        _, rng = bearing(cam, (target.x, target.y))
        info = self.map_info
        return raycast_background(
            self.blocking, (info.origin.position.x, info.origin.position.y), info.resolution,
            (cam.x, cam.y), math.atan2(target.y - cam.y, target.x - cam.x),
            rng + params.depth_through_m + 0.5,
        )

    # ---------- Decision ----------

    def decide(self, object_id, window, window_end):
        target, params, frames = window["target"], window["params"], window["frames"]
        out = window_outcome(frames, params)

        if out.outcome == OUTCOME_PRESENT:
            self.last_resight[object_id] = time.time()
            matches = [m for f in frames for m in f.matches]
            self.confirm_pub.publish(self.object_msg(
                target.class_name,
                float(np.mean([m[0] for m in matches])),
                float(np.mean([m[1] for m in matches])),
                float(np.mean([m[2] for m in matches])),
            ))
        resighted = self.resight_others(window, params)
        if out.outcome == OUTCOME_ABSENT:
            if target.class_name:
                self.remove_pub.publish(self.object_msg(target.class_name, target.x, target.y, target.width))
            else:  # the ledger matches removals by class
                rospy.logwarn("observation_evaluator: %s absent but class unknown; not removing", object_id)

        depth_frames = [f for f in frames if f.depth_mode != "off"]
        res = ObservationResult(
            object_id=object_id,
            class_name=target.class_name,
            outcome=out.outcome,
            reason=out.reason,
            window_start=window["start"],
            window_end=window_end,
            frames=out.frames,
            present_frames=out.present,
            absent_frames=out.absent,
            uncertain_frames=out.uncertain,
            uncertain_reasons=list(out.uncertain_reasons.keys()),
            uncertain_counts=list(out.uncertain_reasons.values()),
            depth_used=params.use_depth,
            depth_modes=sorted({f.depth_mode for f in depth_frames}),
            mean_depth_valid=float(np.mean([f.depth_valid for f in depth_frames])) if depth_frames else 0.0,
            mean_depth_hit=float(np.mean([f.depth_hit for f in depth_frames])) if depth_frames else 0.0,
            mean_depth_through=float(np.mean([f.depth_through for f in depth_frames])) if depth_frames else 0.0,
            mean_depth_near=float(np.mean([f.depth_near for f in depth_frames])) if depth_frames else 0.0,
            resighted_object_ids=resighted,
        )
        res.header.stamp = rospy.Time.now()
        res.header.frame_id = "map"
        self.result_pub.publish(res)
        rospy.loginfo(
            "observation_evaluator: %s (%s) -> %s: %s; uncertain %s; re-sighted %s",
            object_id, target.class_name or "?", ["PRESENT", "ABSENT", "INCONCLUSIVE"][out.outcome],
            out.reason, out.uncertain_reasons or "-", resighted or "-",
        )

    def resight_others(self, window, params):
        """Send re-sightings for other known objects detected in enough frames of the window."""
        sent = []
        now = time.time()
        for oid, (matches, n_frames) in sorted(window["others"].items()):
            if n_frames < params.min_present_frames:
                continue
            if now - self.last_resight.get(oid, -math.inf) < self.resight_min_interval_s:
                continue
            with self.lock:
                known = self.objects.get(oid)
            if known is None:
                continue  # removed meanwhile
            self.last_resight[oid] = now
            self.confirm_pub.publish(self.object_msg(
                known[0],
                float(np.mean([m[0] for m in matches])),
                float(np.mean([m[1] for m in matches])),
                float(np.mean([m[2] for m in matches])),
            ))
            sent.append(oid)
        return sent

    @staticmethod
    def object_msg(class_name, x, y, width):
        msg = DetectedObject(class_name=class_name, width=width)
        msg.pose = Pose()
        msg.pose.position.x, msg.pose.position.y = x, y
        msg.pose.orientation.w = 1.0
        return msg


if __name__ == "__main__":
    ObservationEvaluatorNode()
    rospy.spin()
