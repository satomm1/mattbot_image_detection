#!/usr/bin/env python3
"""Map-frame people tracker: /detected_objects (person) -> /tracked_people (position + velocity).

One constant-velocity Kalman filter per person (people_tracking.tracker). Each
DetectedObjectArray is one frame at its header stamp (the image time), so frames
with no people age the tracks too. Publishes on every frame:

  ~tracked_topic  (/tracked_people)          TrackedPersonArray, confirmed tracks, map frame
  ~marker_topic   (/tracked_people_markers)  MarkerArray: disc, velocity arrow, id (rviz)

Params: ~detections_topic (/detected_objects), ~classes ([person]) plus every
people_tracking.TrackerParams field (accel_std, meas_std, gate_m, min_hits,
max_coast_s, max_speed, ...).
"""

import dataclasses
import math

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from mattbot_image_detection.msg import DetectedObjectArray, TrackedPerson, TrackedPersonArray
from people_tracking import PeopleTracker, TrackerParams


class PeopleTrackerNode:
    def __init__(self):
        defaults = TrackerParams()
        params = TrackerParams(**{f.name: type(getattr(defaults, f.name))(rospy.get_param("~" + f.name,
                                                                                         getattr(defaults, f.name)))
                                  for f in dataclasses.fields(TrackerParams)})
        self.tracker = PeopleTracker(params)
        self.classes = set(rospy.get_param("~classes", ["person"]))
        self.pub = rospy.Publisher(rospy.get_param("~tracked_topic", "/tracked_people"), TrackedPersonArray,
                                   queue_size=5)
        self.marker_pub = rospy.Publisher(rospy.get_param("~marker_topic", "/tracked_people_markers"), MarkerArray,
                                          queue_size=2)
        self._shown_ids = set()
        rospy.Subscriber(rospy.get_param("~detections_topic", "/detected_objects"), DetectedObjectArray,
                         self.detections_callback, queue_size=5)
        rospy.loginfo("people_tracker: tracking %s (%s)", sorted(self.classes), params)

    def detections_callback(self, msg):
        stamp = msg.header.stamp if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now()
        if msg.header.frame_id.lstrip("/") not in ("", "map"):
            rospy.logwarn_throttle(10.0, "people_tracker: detections in frame %r, expected map", msg.header.frame_id)
            return
        people = [o for o in msg.objects if o.class_name in self.classes]
        t = stamp.to_sec()
        self.tracker.update(t, [(o.pose.position.x, o.pose.position.y) for o in people],
                            [o.width for o in people])
        tracks = self.tracker.confirmed(t)
        out = TrackedPersonArray()
        out.header.stamp = stamp
        out.header.frame_id = "map"
        for tr in tracks:
            p = TrackedPerson()
            p.id = tr.id
            p.x, p.y, p.vx, p.vy = (float(v) for v in tr.x)
            p.radius, p.pos_std, p.vel_std = tr.radius, tr.pos_std, tr.vel_std
            p.hits = tr.hits
            p.last_seen = rospy.Time.from_sec(tr.last_seen)
            out.people.append(p)
        self.pub.publish(out)
        if self.marker_pub.get_num_connections():
            self.marker_pub.publish(self._markers(out))

    def _markers(self, arr):
        markers = MarkerArray()
        live = set()
        for p in arr.people:
            live.add(p.id)
            fresh = (arr.header.stamp - p.last_seen).to_sec() < 0.3
            color = ColorRGBA(1.0, 0.4, 0.0, 0.8) if fresh else ColorRGBA(0.6, 0.6, 0.6, 0.6)
            disc = Marker(header=arr.header, ns="people", id=p.id, type=Marker.CYLINDER, action=Marker.ADD)
            disc.pose.position.x, disc.pose.position.y, disc.pose.position.z = p.x, p.y, 0.5
            disc.pose.orientation.w = 1.0
            disc.scale.x = disc.scale.y = 2.0 * p.radius
            disc.scale.z = 1.0
            disc.color = color
            arrow = Marker(header=arr.header, ns="people_velocity", id=p.id, type=Marker.ARROW, action=Marker.ADD)
            arrow.points = [Point(p.x, p.y, 1.1), Point(p.x + p.vx, p.y + p.vy, 1.1)]  # 1 s of motion
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.05, 0.1, 0.1
            arrow.color = ColorRGBA(0.0, 0.6, 1.0, 0.9)
            text = Marker(header=arr.header, ns="people_id", id=p.id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            text.pose.position.x, text.pose.position.y, text.pose.position.z = p.x, p.y, 1.4
            text.pose.orientation.w = 1.0
            text.scale.z = 0.25
            text.color = ColorRGBA(1.0, 1.0, 1.0, 1.0)
            text.text = "%d  %.2f m/s" % (p.id, math.hypot(p.vx, p.vy))
            markers.markers += [disc, arrow, text]
        for gone in self._shown_ids - live:
            for ns in ("people", "people_velocity", "people_id"):
                markers.markers.append(Marker(header=arr.header, ns=ns, id=gone, action=Marker.DELETE))
        self._shown_ids = live
        return markers


if __name__ == "__main__":
    rospy.init_node("people_tracker")
    PeopleTrackerNode()
    rospy.spin()
