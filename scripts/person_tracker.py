import rospy
import numpy as np
from sort import *
import time

from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, Person, PersonArray
from visualization_msgs.msg import Marker, MarkerArray

class PersonTracker:

    def __init__(self):
        self.tracker = Sort()

        self.marker_array = MarkerArray()
        self.person_marker_publisher = rospy.Publisher('/person_marker', MarkerArray, queue_size=10)
        self.person_publisher = rospy.Publisher('/person', PersonArray, queue_size=10)

        self.detection_time = dict()
        self.active_tracks = dict()  # Key: track_id, Value: time since last detection
        self.previous_poses = dict()  # Key: track_id, Value: pose
        self.poses = dict()  # Key: track_id, Value: pose
        self.future_pose1 = dict()  # Key: track_id, Value: future pose 1
        self.future_pose2 = dict()
        self.is_static = dict()  # Key: track_id, Value: is_static

        self.num_detected = 0
        self.object_detect_subscriber = rospy.Subscriber('/detected_objects', DetectedObjectArray, self.detected_objects_callback, queue_size=10)

    def detected_objects_callback(self, msg):
        current_time = time.time()

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

                self.detection_time[person_id] = current_time

                # Add the new track to the active_tracks dictionary
                self.active_tracks[person_id] = 0
                self.previous_poses[person_id] = pose
                self.poses[person_id] = pose
                self.future_pose1[person_id] = pose
                self.future_pose2[person_id] = pose
                self.is_static[person_id] = True

                self.num_detected += 1
            else:
                self.marker_array.markers[int(person_id-1)] = marker
                self.active_tracks[person_id] = 0                

                if np.linalg.norm(np.array(pose) - np.array(self.poses[person_id])) < 0.1:
                    self.is_static[person_id] = True
                    self.future_pose1[person_id] = pose
                    self.future_pose2[person_id] = pose
                else:
                    dt = current_time - self.detection_time[person_id]
                    vx = (pose[0] - self.poses[person_id][0]) / dt
                    vy = (pose[1] - self.poses[person_id][1]) / dt
                    self.is_static[person_id] = False
                    self.future_pose1[person_id] = [pose[0] + vx, pose[1] + vy]
                    self.future_pose2[person_id] = [pose[0] + 2*vx, pose[1] + 2*vy]

                self.detection_time[person_id] = current_time
                self.previous_poses[person_id] = self.poses[person_id]
                self.poses[person_id] = pose

            # Publish the person array
            person_message = Person()
            person_message.id = person_id
            person_message.static = self.is_static[person_id]
            person_message.pose.position.x = pose[0]
            person_message.pose.position.y = pose[1]
            person_message.pose.position.z = 0
            person_message.future_pose1.position.x = self.future_pose1[person_id][0]
            person_message.future_pose1.position.y = self.future_pose1[person_id][1]
            person_message.future_pose1.position.z = 0
            person_message.future_pose2.position.x = self.future_pose2[person_id][0]
            person_message.future_pose2.position.y = self.future_pose2[person_id][1]
            person_message.future_pose2.position.z = 0
            person_message.exited = False

            person_array.persons.append(person_message)

        self.person_publisher.publish(person_array)
        self.person_marker_publisher.publish(self.marker_array)

    def run(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            # Increment the time since last detection for each active track
            person_array = PersonArray()
            keys_to_remove = []
            for track_id in self.active_tracks.keys():
                self.active_tracks[track_id] += 1

                # If the track has not been detected in the last 10 seconds, remove it
                if self.active_tracks[track_id] > 10:
                    self.marker_array.markers[track_id-1].action = Marker.DELETE
                    self.person_marker_publisher.publish(self.marker_array)
                    keys_to_remove.append(track_id)
                    
                    person_message = Person()
                    person_message.id = track_id
                    person_message.static = True
                    person_message.pose.position.x = self.poses[track_id][0]
                    person_message.pose.position.y = self.poses[track_id][1]
                    person_message.pose.position.z = 0
                    person_message.exited = True

                    person_array.persons.append(person_message)

            for key in keys_to_remove:
                self.active_tracks.pop(key)
                self.previous_poses.pop(key)
                self.poses.pop(key)
                self.future_pose1.pop(key)
                self.future_pose2.pop(key)
                self.is_static.pop(key)
                self.detection_time.pop(key)
            
            if len(person_array.persons) > 0:
                self.person_publisher.publish(person_array)

            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('person_tracker', anonymous=True)
    person_tracker = PersonTracker()
    person_tracker.run()