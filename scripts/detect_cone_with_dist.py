import rospy
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3D, Detection3DArray, BoundingBox3D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance, Pose
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray
import message_filters
import tf
from tf.transformations import euler_from_quaternion

from ultralytics import YOLO
import numpy as np
import cv2
import time

import json
import os

def callback(data):
    # Get the image from the message
    image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

    # Perform object detection
    results = model.predict(image, device=0, conf=0.5, agnostic_nms=True)

    # Draw the bounding boxes
    image_with_boxes = results[0].plot()

    # Create the ROS Image and publish it
    image_msg = Image()
    image_msg.data = image_with_boxes.tobytes()
    image_msg.height = image_with_boxes.shape[0]
    image_msg.width = image_with_boxes.shape[1]
    image_msg.encoding = 'rgb8'
    image_msg.step = 3 * image_with_boxes.shape[1]
    image_msg.header.stamp = rospy.Time.now()

    publisher.publish(image_msg)

def unifiedCallback(rgb_data, depth_data):
    start_time = time.time()

    # Get the image from the message
    image = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)

    # Perform object detection
    results = model.predict(image, device=0, conf=0.5, agnostic_nms=True)

    image_time = rgb_data.header.stamp
    
    # Draw the bounding boxes
    image_with_boxes = results[0].plot()

    num_detected = len(results[0].boxes.cls)

    detection_array = DetectedObjectArray()
    detection_array.header.stamp = rospy.Time.now()
    detection_array.header.frame_id = 'map'
    detection_array.objects = []

    if num_detected > 0:
        # Convert depth image to numpy array
        depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)
    
    detected_cone_list = []

    data_dict = {}
    j = 0
    for i in range(num_detected):
        class_num = results[0].boxes.cls[i].item()
        class_name = model.names[class_num]
        class_score = results[0].boxes.conf[i].item()
        
        x_min = results[0].boxes.xyxy[i][0].item()
        y_min = results[0].boxes.xyxy[i][1].item()
        x_max = results[0].boxes.xyxy[i][2].item()
        y_max = results[0].boxes.xyxy[i][3].item()

        x_range = np.arange(x_min, x_max).astype(int)
        y_range = np.arange(y_min, y_max).astype(int)

        x, y = np.meshgrid(x_range, y_range)

        x = x.flatten()
        y = y.flatten()

        X = (x - cx) * depth[y, x] / fx
        Y = (y - cy) * depth[y, x] / fy

        print(X)

        print('Class: {}, Score: {:.2f}, x_min: {:.2f}, y_min: {:.2f}, x_max: {:.2f}, y_max: {:.2f}'.format(class_name, class_score, x_min, y_min, x_max, y_max))
        
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
            print('Estimated depth of {} is {} meters'.format(class_name, estimated_depth))


            # Get map to camera_link transform
            (trans, rot) = tf_listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
            x_camera = trans[0]
            y_camera = trans[1]
            _, _, theta_camera = euler_from_quaternion(rot)
            theta_camera = theta_camera
            print("Theta camera: ", theta_camera)

            x_min = np.min(X)
            y_min = np.min(Y)
            x_max = np.max(X)
            y_max = np.max(Y)

            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2

            hypot = np.sqrt(x_center**2 + estimated_depth**2)

            theta_object = np.arctan2(x_center, estimated_depth)
            x_map = x_camera + hypot * np.cos(theta_camera - theta_object)
            y_map = y_camera + hypot * np.sin(theta_camera - theta_object)

            object_dict = {}
            object_dict['class_name'] = class_name
            object_dict['depth'] = estimated_depth
            object_dict['x_min'] = x_min
            object_dict['y_min'] = y_min
            object_dict['x_max'] = x_max
            object_dict['y_max'] = y_max
            object_dict['x_map'] = x_map
            object_dict['y_map'] = y_map
            data_dict[j] = object_dict
            j += 1

            detected_cone_list.append([x[0], y[0], np.max(x), np.max(y), class_score])

            # Create DetectedObject message
            detected_object = DetectedObject()
            detected_object.class_name = class_name
            detected_object.probability = 1.0
            detected_object_pose = Pose()
            detected_object_pose.position.x = x_map
            detected_object_pose.position.y = y_map
            detected_object_pose.position.z = 0
            detected_object_pose.orientation.w = 1
            detected_object.pose = detected_object_pose
            detected_object.width = x_max - x_min
            detection_array.objects.append(detected_object)

            # # Write estimated depth on the image
            image_with_boxes = cv2.putText(image_with_boxes, '{}: {:.2f} m'.format(class_name, estimated_depth), (x[0]+10, y[0]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            image_with_boxes = cv2.putText(image_with_boxes, '{:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(x_min, y_min, x_max, y_max), (x[0]+10, y[0]+70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            image_with_boxes = cv2.putText(image_with_boxes, '{:.2f}, {:.2f}'.format(x_map, y_map), (x[0]+10, y[0]+90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


    # Create the ROS Image and publish it
    image_msg = Image()
    image_msg.data = image_with_boxes.tobytes()
    image_msg.height = image_with_boxes.shape[0]
    image_msg.width = image_with_boxes.shape[1]
    image_msg.encoding = 'rgb8'
    image_msg.step = 3 * image_with_boxes.shape[1]
    image_msg.header.stamp = rospy.Time.now()
    publisher.publish(image_msg)

    # Publish the detected objects array if any objects exist
    if len(detection_array.objects) > 0:
        detected_object_publisher.publish(detection_array)      

    end_time = time.time()
    print('Elapsed time: {}'.format(end_time - start_time))


if __name__ == '__main__':
    
    # Initialize the ROS node
    rospy.init_node('cone_detection', anonymous=True)

    weights_file = rospy.get_param('~weights_file', 'cone.pt')
    model = YOLO(weights_file)


    # Create the publisher that will show image with bounding boxes
    publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)
    detected_object_publisher = rospy.Publisher('/detected_objects', DetectedObjectArray, queue_size=10)

    depth_readings = []
    depth_timestamps = []

    camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
    camera_info = camera_info_msg.K
    camera_info = np.array(camera_info).reshape(3, 3)
    fx = camera_info[0, 0]
    fy = camera_info[1, 1]
    cx = camera_info[0, 2]
    cy = camera_info[1, 2]

    tf_listener = tf.TransformListener()

    rbg_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)

    ts = message_filters.ApproximateTimeSynchronizer([rbg_sub, depth_sub], 1, 0.1)
    ts.registerCallback(unifiedCallback)

    rospy.spin()