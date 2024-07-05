import rospy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3D, Detection3DArray, BoundingBox3D, ObjectHypothesisWithPose
from geometry_msgs.msg import PoseWithCovariance
import message_filters

from ultralytics import YOLO
import numpy as np
import cv2

def depth_callback(data):
    # Convert to numpy array
    depth = np.frombuffer(data.data, dtype=np.uint16).reshape(data.height, data.width)

    # Append the depth reading and timestamp to the list
    depth_readings.append(depth)
    depth_timestamps.append(data.header.stamp)

    # Remove the oldest reading if the list is too long
    if (len(depth_readings) > 5):
        depth_readings.pop(0)
        depth_timestamps.pop(0)


def image_callback(data):
    # Get the image from the message
    image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

    # Perform object detection
    results = model.predict(image, device=0, conf=0.6, agnostic_nms=True)

    image_time = data.header.stamp
    
    # Find depth timestamp that is closest to the image timestamp
    closest_time = min(depth_timestamps, key=lambda x: abs(x - image_time))
    index = depth_timestamps.index(closest_time)

    # Get the depth reading that corresponds to the image and take the mask
    depth = depth_readings[index]

    # print(results[0].boxes)
    # print(results[0].masks.xy[0])

    # Draw the bounding boxes
    image_with_boxes = results[0].plot()

    num_detected = len(results[0].boxes.cls)

    detection_array = Detection3DArray()
    detection_array.header.stamp = rospy.Time.now()
    detection_array.header.frame_id = 'camera_link'

    for i in range(num_detected):
        class_num = results[0].boxes.cls[i].item()
        class_name = model.names[class_num]
        
        x = results[0].masks.xy[i][:, 0].astype(int)
        y = results[0].masks.xy[i][:, 1].astype(int)

        depth_values = depth[y, x]

        # Remove zero depth values
        indx = np.where(depth_values > 0)
        depth_values = depth_values[indx]

        estimated_depth = np.mean(depth_values)/1000  # meters
        print('Estimated depth of {} is {} meters'.format(class_name, estimated_depth))

        # Write estimated depth on the image
        image_with_boxes = cv2.putText(image_with_boxes, '{}: {:.2f} m'.format(class_name, estimated_depth), (x[0]+10, y[0]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # detection = Detection3D()
        # detection.results.hypothesis.class_id = class_num
        # detection.results.hypothesis.score = results[0].boxes.conf[i].item()

        # detected_pose = PoseWithCovariance()
        # TODO...

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
    # Get the image from the message
    image = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)

    # Perform object detection
    results = model.predict(image, device=0, conf=0.6, agnostic_nms=True)

    image_time = rgb_data.header.stamp
    
    # Draw the bounding boxes
    image_with_boxes = results[0].plot()

    num_detected = len(results[0].boxes.cls)

    if num_detected > 0:
        # Convert depth image to numpy array
        depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)

    # detection_array = Detection3DArray()
    # detection_array.header.stamp = rospy.Time.now()
    # detection_array.header.frame_id = 'camera_link'

    for i in range(num_detected):
        class_num = results[0].boxes.cls[i].item()
        class_name = model.names[class_num]
        
        x = results[0].masks.xy[i][:, 0].astype(int)
        y = results[0].masks.xy[i][:, 1].astype(int)

        depth_values = depth[y, x]

        # Remove zero depth values
        indx = np.where(depth_values > 0)
        depth_values = depth_values[indx]

        estimated_depth = np.mean(depth_values)/1000  # meters
        print('Estimated depth of {} is {} meters'.format(class_name, estimated_depth))

        # Write estimated depth on the image
        image_with_boxes = cv2.putText(image_with_boxes, '{}: {:.2f} m'.format(class_name, estimated_depth), (x[0]+10, y[0]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # detection = Detection3D()
        # detection.results.hypothesis.class_id = class_num

    # Create the ROS Image and publish it
    image_msg = Image()
    image_msg.data = image_with_boxes.tobytes()
    image_msg.height = image_with_boxes.shape[0]
    image_msg.width = image_with_boxes.shape[1]
    image_msg.encoding = 'rgb8'
    image_msg.step = 3 * image_with_boxes.shape[1]
    image_msg.header.stamp = rospy.Time.now()
    publisher.publish(image_msg)

if __name__ == '__main__':
    # Initialize the ROS node
    rospy.init_node('image_detection', anonymous=True)

    # self.K = np.array([[570.3405082258201,               0.0, 319.5],
    #                        [0.0,               570.3405082258201, 239.5],
    #                        [0.0,                             0.0,   1.0]])
    # self.R = np.identity(3)
    # self.P = np.array([[570.3405082258201,               0.0, 319.5, 0.0],
    #                     [              0.0, 570.3405082258201, 239.5, 0.0],
    #                     [              0.0,               0.0,   1.0, 0.0]])
                        
    # self.K_inv = np.linalg.inv(self.K)
    # self.P_inv = np.linalg.inv(self.P[:, :3])
    
    # self.fx = 570.3405082258201
    # self.fy = 570.3405082258201
    # self.S = 0
    # self.cx = 319.5
    # self.cy = 239.5
    
    # self.Converter = np.array([[1/self.fx, -self.S/(self.fx * self.fy), (self.S*self.cy - self.cx*self.fy)/(self.fx*self.fy)],
    #                             [0, 1/self.fy, -self.cy/self.fy],
    #                             [0, 0, 1]])

    weights_file = rospy.get_param('~weights_file', 'yolov8n-seg.pt')
    model = YOLO(weights_file)

    # Create the publisher that will show image with bounding boxes
    publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)
    # detected_object_publisher = rospy.Publisher('/detected_objects', Detection2DArray, queue_size=10)

    depth_readings = []
    depth_timestamps = []

    # Subscribe to the image topic
    # rospy.Subscriber('/camera/depth/image_raw', Image, depth_callback, queue_size=1)
    # rospy.Subscriber('/camera/color/image_raw', Image, image_callback, queue_size=1)

    rbg_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
    depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)

    ts = message_filters.ApproximateTimeSynchronizer([rbg_sub, depth_sub], 1, 0.1)
    ts.registerCallback(unifiedCallback)

    rospy.spin()
