import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
import message_filters
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import tf
from tf.transformations import euler_from_quaternion

from ultralytics import YOLO
import numpy as np
import cv2
import time

import json
import os

IOU_THRESHOLD = 0.1  # Set the IoU threshold for NMS
OBJECT_CONFIDENCE_THRESHOLD = 0.7
OTHER_CONFIDENCE_THRESHOLD = 0.03
COLORS = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'yellow']
COLOR_CODES = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128), (255, 192, 203), (255, 165, 0), (255, 255, 0)]

class StochOccupancyGrid2D(object):
    def __init__(self, resolution, width, height, origin_x, origin_y,
                window_size, probs, thresh=0.5, robot_d=0.6):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = np.reshape(np.asarray(probs), (height, width))
        self.l = np.zeros((height, width))
        self.window_size = window_size # window_size
        # print(window_size)
        self.thresh = thresh
        self.robot_d=robot_d

    def snap_to_grid(self, x):
        return (self.resolution*round(x[0]/self.resolution), self.resolution*round(x[1]/self.resolution))

    def snap_to_grid1(self, x):
        return (self.resolution * np.round(x[0] / self.resolution), self.resolution * np.round(x[1] / self.resolution))

    def get_index(self, x):
        return (np.round((x[0]-self.origin_x)/self.resolution), np.round((x[1]-self.origin_y)/self.resolution))

    def add_to_map(self, x, y, width):
        x1, y1 = self.get_index((x, y))
        r = width / 2

        x_min = int(max(0, x1 - r / self.resolution))
        x_max = int(min(self.width - 1, x1 + r / self.resolution))
        y_min = int(max(0, y1 - r / self.resolution))
        y_max = int(min(self.height - 1, y1 + r / self.resolution))

        self.probs[y_min:y_max+1, x_min:x_max+1] = 1.0

    def is_overlapped(self, x, y, width, height):
        # Given global x/y coordinates, and width/height of the object in meters, return if the object is overlapped with the map
        x1, y1 = self.get_index((x, y))

        # Take minimum of width/height as the object's radius
        r = min(width, height) / 2

        # Check if the object is within the map
        if x1 < 0 or y1 < 0 or x1 >= self.width or y1 >= self.height:
            return True
        
        # Calculate the grid indices of the object's bounding box
        x_min = int(max(0, x1 - r / self.resolution))
        x_max = int(min(self.width - 1, x1 + r / self.resolution))
        y_min = int(max(0, y1 - r / self.resolution))
        y_max = int(min(self.height - 1, y1 + r / self.resolution))

        # Count the number of occupied cells within the bounding box
        occupied_cells = 0
        total_cells = (x_max - x_min + 1) * (y_max - y_min + 1)
        x_all, y_all = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        occupied = np.where(self.probs[y_all, x_all] > self.thresh)
        occupied_cells = len(occupied[0])

        # Calculate the percentage of occupied cells
        occupied_percentage = (occupied_cells / total_cells) * 100

        return occupied_percentage > 20  # Return True if more than 50% of the cells are occupied


class ConeDetector:
    
    def __init__(self):
        print("TESTING ************************")
        weights_file = rospy.get_param('~weights_file', '../weights/cone_person.pt')
        self.object_model = YOLO(weights_file)

        print(weights_file)

        coco_labels_file = rospy.get_param('~coco_labels_file', 'yolov8_labels.txt')
        print(coco_labels_file)
        self.other_model = YOLO('yolov8n.pt')
        with open(coco_labels_file, 'r') as f:
            self.coco_labels = f.read().splitlines()

        self.is_turning = False
        self.time_since_turning = 0

        # Create the publisher that will show image with bounding boxes
        self.publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)
        self.detected_object_publisher = rospy.Publisher('/detected_objects', DetectedObjectArray, queue_size=10)
        self.unknown_object_publisher = rospy.Publisher('/unknown_objects', DetectedObjectWithImageArray, queue_size=10)
        self.marker_pub = rospy.Publisher('/text_marker', MarkerArray, queue_size=10)

        self.num_markers = 0
        self.marker_array = MarkerArray()

        camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        camera_info = camera_info_msg.K
        camera_info = np.array(camera_info).reshape(3, 3)
        self.fx = camera_info[0, 0]
        self.fy = camera_info[1, 1]
        self.cx = camera_info[0, 2]
        self.cy = camera_info[1, 2]

        self.map_msg = rospy.wait_for_message("/navigation_map", OccupancyGrid)
        self.map = StochOccupancyGrid2D(self.map_msg.info.resolution, 
                                             self.map_msg.info.width, 
                                            self.map_msg.info.height,
                                 self.map_msg.info.origin.position.x,
                                 self.map_msg.info.origin.position.y,
                                                                   5,
                                                   self.map_msg.data)

        self.tf_listener = tf.TransformListener()

        self.unknown_label_subscriber = rospy.Subscriber('/labeled_unknown_objects', DetectedObjectArray, self.unknown_label_callback, queue_size=10)
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

        unknown_object_array = DetectedObjectWithImageArray()
        unknown_object_array.header.stamp = rospy.Time.now()
        unknown_object_array.header.frame_id = 'map'
        unknown_object_array.objects = []

        detected_cone_list = []

        data_dict = {}
        data_count = 0

        if num_detected > 0:
            # Convert depth image to numpy array
            depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)

            (detected_cone_list, 
            data_dict, 
            data_count, 
            detection_array, 
            unknown_object_array,
            image_with_boxes,
            image) = self.depth_calculation(object_results, num_detected, depth, trans, rot, 
                                            detected_cone_list, data_dict, data_count, detection_array, 
                                            unknown_object_array, image_with_boxes, image)
        if num_detected_other > 0 and num_detected == 0:
            # Convert depth image to numpy array

            depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)

            (detected_cone_list, 
            data_dict, 
            data_count, 
            detection_array, 
            unknown_object_array,
            image_with_boxes,
            image) = self.depth_calculation(other_results, num_detected_other, depth, trans, rot, 
                                                detected_cone_list, data_dict, data_count, detection_array, 
                                                unknown_object_array, image_with_boxes, image, coco=True)
        elif num_detected_other > 0:
            (detected_cone_list, 
            data_dict, 
            data_count, 
            detection_array, 
            unknown_object_array,
            image_with_boxes,
            image) = self.depth_calculation(other_results, num_detected_other, depth, trans, rot, 
                                                detected_cone_list, data_dict, data_count, detection_array, 
                                                unknown_object_array, image_with_boxes, image, coco=True)

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

        # Publish the unknown objects array if any objects exist
        if len(unknown_object_array.objects) > 0:   
            _, buffer = cv2.imencode('.jpg', image)
            unknown_object_array.data = np.array(buffer).tobytes()
            self.unknown_object_publisher.publish(unknown_object_array)

        end_time = time.time()
        print('Detection time: {}'.format(detect_time - start_time))
        print('Elapsed time: {}'.format(end_time - start_time))

    def depth_calculation(self, results, num_detected, depth, trans, rot, detected_cone_list, data_dict, data_count, detection_array, 
                           unknown_object_array, image_with_boxes, image, coco=False):
        # Get map to camera_link transform
        x_camera = trans[0]
        y_camera = trans[1]
        _, _, theta_camera = euler_from_quaternion(rot)
        theta_camera = theta_camera
        
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

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Get the estimated depth of the object (in meters)
            estimated_depth = self.get_depth(x1, y1, x2, y2, depth)

            if estimated_depth == 0 or estimated_depth > 5:
                continue

            # Get the object's position in the camera frame
            local_x_min = (x1 - self.cx) * estimated_depth / self.fx  # meters
            local_z_min = (y1 - self.cy) * estimated_depth / self.fy  # meters
            local_x_max = (x2 - self.cx) * estimated_depth / self.fx  # meters
            local_z_max = (y2 - self.cy) * estimated_depth / self.fy  # meters
            local_x_center = (x_center - self.cx) * estimated_depth / self.fx  # meters
            local_z_center = (y_center - self.cy) * estimated_depth / self.fy  # meters

            # Calculate the object's width and height
            object_width = local_x_max - local_x_min
            object_height = local_z_max - local_z_min
            

            if object_height < 0.1 or object_width < 0.1:
                continue  # Skip objects that are too small
            object_depth = np.min([object_width, object_height, 0.33])
            
            hypot = np.sqrt(local_x_center**2 + (estimated_depth + object_depth/2)**2)

            # Calculate the object's position in the map frame
            theta_object = np.arctan2(local_x_center, estimated_depth)
            x_map = x_camera + (hypot + object_width/2) * np.cos(theta_camera - theta_object)  # Add object width/2 to get center of object (assume width=depth)
            y_map = y_camera + (hypot + object_width/2) * np.sin(theta_camera - theta_object)

            is_overlapped = False
            if self.map.is_overlapped(x_map, y_map, object_width, object_height):
                # image_with_boxes = cv2.putText(image_with_boxes, 'Overlapped', (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                is_overlapped = True

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
                detected_cone_list.append([x1, y1, x2, y2, class_score])

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

            if not is_overlapped and class_name == 'unknown':
                if len(unknown_object_array.objects) < len(COLORS):

                    # Get the part of the image within the bounding box
                    object_image = image[y1:y2, x1:x2]
                    _, buffer = cv2.imencode('.jpg', object_image)
                    unknown_object = DetectedObjectWithImage()
                    unknown_object.class_name = class_name
                    unknown_object.probability = class_score
                    unknown_object.pose = detected_object_pose
                    unknown_object.width = object_width
                    unknown_object.x1 = x1
                    unknown_object.y1 = y1
                    unknown_object.x2 = x2
                    unknown_object.y2 = y2
                    unknown_object.color = COLORS[len(unknown_object_array.objects)]
                    # Include the bounding box image in the message
                    unknown_object.data = np.array(buffer)[x1:x2, y1:y2].tobytes()
                    image = cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_CODES[len(unknown_object_array.objects)], 2)
                    # unknown_object.data = np.array(buffer).tobytes()
                    unknown_object_array.objects.append(unknown_object)

            # Write estimated depth on the image
            image_with_boxes = cv2.putText(image_with_boxes, 'Depth: {:.2f} m'.format(estimated_depth), (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # image_with_boxes = cv2.putText(image_with_boxes, '{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(x_min, y_min, x_max, y_max), (x[0]+10, y[0]+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # image_with_boxes = cv2.putText(image_with_boxes, '{:.2f}, {:.2f}'.format(x_map, y_map), (x1+10, y1+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            # image_with_boxes = cv2.putText(image_with_boxes, 'Width: {:.2f} m'.format(object_width), (x1+10, y1+110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


            if coco:  # Draw bounding box and label for COCO classes
                if class_name == "unknown":
                    if is_overlapped:
                        image_with_boxes = cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    else:
                        image_with_boxes = cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    image_with_boxes = cv2.putText(image_with_boxes, '{}: {:.2f}'.format(class_name, class_score), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                else:
                    if is_overlapped:
                        image_with_boxes = cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    else:
                        image_with_boxes = cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), (126, 0, 126), 2)
                        self.map.add_to_map(x_map, y_map, object_width)
                        self.publish_text(x_map, y_map, class_name)
                    image_with_boxes = cv2.putText(image_with_boxes, '{}: {:.2f}'.format(class_name, class_score), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        return detected_cone_list, data_dict, data_count, detection_array, unknown_object_array, image_with_boxes, image

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

    def unknown_label_callback(self, msg):
        # For each object, add it to the map
        for obj in msg.objects:
            # Get the object's position in the map frame
            x_map = obj.pose.position.x
            y_map = obj.pose.position.y
            object_width = obj.width

            # Add the object to the map
            self.map.add_to_map(x_map, y_map, object_width)
            self.publish_text(x_map, y_map, obj.class_name)

    def publish_text(self, x, y, class_name):
        # Publish to RVIZ
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = rospy.Time.now()
        marker.ns = "text"
        marker.id = self.num_markers
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.text = class_name

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.5
        marker.pose.orientation.w = 1.0

        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0

        self.marker_array.markers.append(marker)

        self.marker_pub.publish(self.marker_array)
        self.num_markers += 1

    def run(self):
        rospy.spin()

time_since_turning = 0
if __name__ == '__main__':
    
    # Initialize the ROS node
    rospy.init_node('cone_detection', anonymous=True)

    detector = ConeDetector()
    detector.run()
    