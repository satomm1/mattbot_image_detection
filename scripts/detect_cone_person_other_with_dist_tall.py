import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray
import message_filters
import tf
from tf.transformations import euler_from_quaternion
import sensor_msgs.point_cloud2 as pc2

from ultralytics import YOLO
import numpy as np
import cv2
import time

import json
import os

IOU_THRESHOLD = 0.1  # Set the IoU threshold for NMS
OBJECT_CONFIDENCE_THRESHOLD = 0.75
OTHER_CONFIDENCE_THRESHOLD = 0.04

class ConeDetector:
    
    def __init__(self):
        weights_file = rospy.get_param('~weights_file', '../weights/cone_person.pt')
        self.object_model = YOLO(weights_file)

        self.other_model = YOLO('yolov8n.pt')
        with open('yolo_coco_labels.txt', 'r') as f:
            self.coco_labels = f.read().splitlines()

        self.is_turning = False
        self.time_since_turning = 0

        # Create the publisher that will show image with bounding boxes
        self.publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)
        self.detected_object_publisher = rospy.Publisher('/detected_objects', DetectedObjectArray, queue_size=10)

        camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        camera_info = camera_info_msg.K
        camera_info = np.array(camera_info).reshape(3, 3)
        self.fx = camera_info[0, 0]
        self.fy = camera_info[1, 1]
        self.cx = camera_info[0, 2]
        self.cy = camera_info[1, 2]

        self.tf_listener = tf.TransformListener()

        self.cmd_vel_subscriber = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)    
        self.rbg_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)

        self.ts = message_filters.ApproximateTimeSynchronizer([self.rbg_sub, self.depth_sub], 1, 0.1)
        self.ts.registerCallback(self.unifiedCallback)

    def unifiedCallback(self, rgb_data, depth_data):

        if self.is_turning:
            # Can't rely on data if robot is turning
            return

        start_time = time.time()

        # Get the transform from map to camera_link
        try:
            (trans, rot) = self.tf_listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        # Get the image from the message
        image = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)

        # Rotate image 180 degrees
        image = cv2.rotate(image, cv2.ROTATE_180)

        # Perform object detection
        object_results = self.object_model.predict(image, device=0, conf=OBJECT_CONFIDENCE_THRESHOLD, agnostic_nms=True, iou=IOU_THRESHOLD)
        other_results = self.other_model.predict(image, device=0, conf=OTHER_CONFIDENCE_THRESHOLD, agnostic_nms=True, iou=IOU_THRESHOLD)
        detect_time = time.time()

        image_time = rgb_data.header.stamp
        
        # Draw the bounding boxes
        image_with_boxes = object_results[0].plot()

        num_detected = len(object_results[0].boxes.cls)
        num_detected_other = len(other_results[0].boxes.cls)

        detection_array = DetectedObjectArray()
        detection_array.header.stamp = rospy.Time.now()
        detection_array.header.frame_id = 'map'
        detection_array.objects = []

        detected_cone_list = []

        data_dict = {}
        data_count = 0

        if num_detected > 0:
            # Convert depth image to numpy array
            depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)

            # Rotate the depth image 180 degrees
            depth = np.flip(depth)
        
            (detected_cone_list, 
            data_dict, 
            data_count, 
            detection_array, 
            image_with_boxes) = self.depth_calculation(object_results, num_detected, depth, trans, rot, 
                                            detected_cone_list, data_dict, data_count, detection_array, 
                                            image_with_boxes)
        if num_detected_other > 0 and num_detected == 0:
            # Convert depth image to numpy array

            depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)

            # Rotate the depth image 180 degrees
            depth = np.flip(depth)
        
            (detected_cone_list, 
            data_dict, 
            data_count, 
            detection_array, 
            image_with_boxes) = self.depth_calculation(other_results, num_detected_other, depth, trans, rot, 
                                                detected_cone_list, data_dict, data_count, detection_array, 
                                                image_with_boxes, coco=True)
        elif num_detected_other > 0:
            (detected_cone_list, 
            data_dict, 
            data_count, 
            detection_array, 
            image_with_boxes) = self.depth_calculation(other_results, num_detected_other, depth, trans, rot, 
                                                detected_cone_list, data_dict, data_count, detection_array, 
                                                image_with_boxes, coco=True)

        # Create the ROS Image and publish it
        image_msg = Image()
        image_msg.data = image_with_boxes.tobytes()
        image_msg.height = image_with_boxes.shape[0]
        image_msg.width = image_with_boxes.shape[1]
        image_msg.encoding = 'rgb8'
        image_msg.step = 3 * image_with_boxes.shape[1]
        image_msg.header.stamp = rospy.Time.now()
        self.publisher.publish(image_msg)

        # Publish the detected objects array if any objects exist
        if len(detection_array.objects) > 0:
            self.detected_object_publisher.publish(detection_array)      

        end_time = time.time()
        print('Detection time: {}'.format(detect_time - start_time))
        print('Elapsed time: {}'.format(end_time - start_time))

    def depth_calculation(self, results, num_detected, depth, trans, rot, detected_cone_list, data_dict, data_count, detection_array, image_with_boxes, coco=False):
        for i in range(num_detected):
            class_num = int(results[0].boxes.cls[i].item())

            if coco:
                class_name = self.coco_labels[class_num]
                if class_name == 'person':
                    continue

            else:
                class_name = self.object_model.names[class_num]
            class_score = results[0].boxes.conf[i].item()

            if class_score < OBJECT_CONFIDENCE_THRESHOLD:
                class_name = 'unknown'
            
            x_min = results[0].boxes.xyxy[i][0].item()
            y_min = results[0].boxes.xyxy[i][1].item()
            x_max = results[0].boxes.xyxy[i][2].item()
            y_max = results[0].boxes.xyxy[i][3].item()

            x1 = int(x_min)
            y1 = int(y_min)
            x2 = int(x_max)
            y2 = int(y_max)

            x_range = np.arange(x_min, x_max).astype(int)
            y_range = np.arange(y_min, y_max).astype(int)

            x, y = np.meshgrid(x_range, y_range)

            # estimated_depth = self.get_depth(x1, y1, x2, y2, depth)

            x = x.flatten()
            y = y.flatten()

            X = (x - self.cx) * depth[y, x] / self.fx
            Y = (y - self.cy) * depth[y, x] / self.fy

            # rospy.loginfo('Class: {}, Score: {:.2f}, x_min: {:.2f}, y_min: {:.2f}, x_max: {:.2f}, y_max: {:.2f}'.format(class_name, class_score, x_min, y_min, x_max, y_max))
            # print('Class: {}, Score: {:.2f}, x_min: {:.2f}, y_min: {:.2f}, x_max: {:.2f}, y_max: {:.2f}'.format(class_name, class_score, x_min, y_min, x_max, y_max))
            
            depth_values = depth[y, x]

            # Remove zero depth values
            indx = np.where(depth_values > 0)

            if len(indx[0]) == 0:
                object_dict = {}
                object_dict['class_name'] = class_name
                object_dict['depth'] = 0.0
                object_dict['x_min'] = 0.0
                object_dict['y_min'] = 0.0
                object_dict['x_max'] = 0.0
                object_dict['y_max'] = 0.0
                object_dict['x_map'] = 0.0
                object_dict['y_map'] = 0.0
            else:
                depth_values = depth_values[indx]
                X = X[indx]/1000  # meters
                Y = Y[indx]/1000  # meters

                # Take the 25th percentile depth
                estimated_depth = np.percentile(depth_values, 25)/1000  # meters
                # print('Estimated depth of {} is {} meters'.format(class_name, estimated_depth))

                if estimated_depth > 5:
                    continue

                # Get map to camera_link transform
                x_camera = trans[0]
                y_camera = trans[1]
                _, _, theta_camera = euler_from_quaternion(rot)
                theta_camera = theta_camera
                # print("Theta camera: ", theta_camera)

                x_min = np.min(X)
                y_min = np.min(Y)
                x_max = np.max(X)
                y_max = np.max(Y)
                object_width = x_max - x_min

                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                x_min = results[0].boxes.xyxy[i][0].item()
                x_max = results[0].boxes.xyxy[i][2].item()

                x_center = ((x_max+x_min)/2 - self.cx) * estimated_depth / self.fx
                y_center = (y - self.cy) * estimated_depth / self.fy

                object_width = (x_max- self.cx) * estimated_depth / self.fx - (x_min - self.cx) * estimated_depth / self.fx

                hypot = np.sqrt(x_center**2 + estimated_depth**2)
                

                theta_object = np.arctan2(x_center, estimated_depth)
                x_map = x_camera + (hypot + object_width/2) * np.cos(theta_camera - theta_object)  # Add object width/2 to get center of object (assume width=depth)
                y_map = y_camera + (hypot + object_width/2) * np.sin(theta_camera - theta_object)

                object_dict = {}
                object_dict['class_name'] = class_name
                object_dict['depth'] = estimated_depth
                object_dict['x_min'] = x_min
                object_dict['y_min'] = y_min
                object_dict['x_max'] = x_max
                object_dict['y_max'] = y_max
                object_dict['x_map'] = x_map
                object_dict['y_map'] = y_map
                data_dict[data_count] = object_dict
                data_count += 1

                if class_name == 'cone':
                    detected_cone_list.append([x[0], y[0], np.max(x), np.max(y), class_score])

                # Create DetectedObject message
                detected_object = DetectedObject()
                detected_object.class_name = class_name
                detected_object.probability = class_score
                detected_object_pose = Pose()
                detected_object_pose.position.x = x_map
                detected_object_pose.position.y = y_map
                detected_object_pose.position.z = 0
                detected_object_pose.orientation.w = 1
                detected_object.pose = detected_object_pose
                detected_object.width = object_width
                detected_object.x1 = x1
                detected_object.y1 = y1
                detected_object.x2 = x2
                detected_object.y2 = y2
                detection_array.objects.append(detected_object)

                # Write estimated depth on the image
                image_with_boxes = cv2.putText(image_with_boxes, '{}: {:.2f} m'.format(class_name, estimated_depth), (x[0]+10, y[0]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                image_with_boxes = cv2.putText(image_with_boxes, '{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(x_min, y_min, x_max, y_max), (x[0]+10, y[0]+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                image_with_boxes = cv2.putText(image_with_boxes, '{:.2f}, {:.2f}'.format(x_map, y_map), (x[0]+10, y[0]+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                image_with_boxes = cv2.putText(image_with_boxes, 'Width: {:.2f}'.format(object_width), (x[0]+10, y[0]+110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

                # TODO draw bounding box if coco
                if coco:
                    image_with_boxes = cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return detected_cone_list, data_dict, data_count, detection_array, image_with_boxes

    def get_depth(self, x_min, y_min, x_max, y_max, depth):
        depth_values = depth[int(y_min):int(y_max), int(x_min):int(x_max)].flatten()

        # Remove zero depth values
        depth_values = depth_values[depth_values > 0]

        if len(depth_values) == 0:
            return 0.0  # or some default value if no valid depth values are found

        # Get the 25th percentile depth
        estimated_depth = np.percentile(depth_values, 25) / 1000.0  # Convert to meters

        return estimated_depth

    def cmd_vel_callback(self, msg):

        if np.abs(msg.angular.z) > 0.07:
            self.is_turning = True
            self.time_since_turning = time.time()
        elif time.time() - self.time_since_turning < 0.25:
            is_turning = True
        else:
            self.is_turning = False

    def run(self):
        rospy.spin()

time_since_turning = 0
if __name__ == '__main__':
    
    # Initialize the ROS node
    rospy.init_node('cone_detection', anonymous=True)

    detector = ConeDetector()
    detector.run()
    