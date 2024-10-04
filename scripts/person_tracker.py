import rospy
import numpy as np
from sort import *

from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, Person, PersonArray
from visualization_msgs.msg import Marker, MarkerArray

class PersonTracker:

    def __init__(self):
        self.tracker = Sort()

        self.marker_array = MarkerArray()
        self.person_marker_publisher = rospy.Publisher('/person_marker', MarkerArray, queue_size=10)
        self.person_publisher = rospy.Publisher('/person', PersonArray, queue_size=10)

        self.active_tracks = dict()  # Key: track_id, Value: time since last detection
        self.poses = dict()  # Key: track_id, Value: pose

        self.num_detected = 0
        self.object_detect_subscriber = rospy.Subscriber('/detected_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=10)

    def detected_objects_callback(self, msg):
        object_array = msg.objects

        detections = []
        poses = []
        for obj in object_array:
            if obj.class_name == "person":
                detections.append([obj.x1, obj.y1, obj.x2, obj.y2, obj.probability])
                poses.append([obj.pose.position.x, obj.pose.position.y])

        detections = np.array(detections)
        track_bbs_ids = self.tracker.update(detections)

        # Ensure that the track_bbs_ids array is sorted by track_id
        sorted_indx = np.argsort(track_bbs_ids[:, 4])
        track_bbs_ids = track_bbs_ids[sorted_indx, :]

        person_array = PersonArray()

        # For visualization purposes
        for ii in range(len(track_bbs_ids[:, 4])):
            print(track_bbs_ids[ii, 4])
            print(poses[ii])
            print("************")

            person_id = int(track_bbs_ids[ii, 4])
            pose = poses[ii]

            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = rospy.Time.now()
            marker.ns = "person_tracker"
            marker.id = person_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = pose[0]
            marker.pose.position.y = pose[1]
            marker.pose.position.z = 0
            marker.pose.orientation.x = 0
            marker.pose.orientation.y = 0
            marker.pose.orientation.z = 0
            marker.pose.orientation.w = 1
            marker.scale.x = 0.1
            marker.scale.y = 0.1
            marker.scale.z = 0.1
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            if person_id > self.num_detected:
                self.marker_array.markers.append(marker)

                # Add the new track to the active_tracks dictionary
                self.active_tracks[person_id] = 0
                self.poses[person_id] = pose

                self.num_detected += 1
            else:
                self.marker_array.markers[int(person_id-1)] = marker
                self.active_tracks[person_id] = 0
                self.poses[person_id] = pose

            # Publish the person array
            person_message = Person()
            person_message.id = person_id
            person_message.static = False # FIXME
            person_message.pose.position.x = pose[0]
            person_message.pose.position.y = pose[1]
            person_message.pose.position.z = 0

            person_array.persons.append(person_message)

        self.person_publisher.publish(person_array)
        self.person_tracker_publisher.publish(self.marker_array)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # Increment the time since last detection for each active track
            for track_id in self.active_tracks.keys():
                self.active_tracks[track_id] += 1

                # If the track has not been detected in the last 10 seconds, remove it
                if self.active_tracks[track_id] > 10:
                    self.marker_array.markers[track_id-1].action = Marker.DELETE
                    self.person_tracker_publisher.publish(self.marker_array)
                    self.active_tracks.pop(track_id)

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('person_tracker', anonymous=True)
    person_tracker = PersonTracker()
    rospy.spin()