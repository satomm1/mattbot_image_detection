import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from image_detection_with_unknowns.msg import LabeledObject, LabeledObjectArray
import message_filters
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Point

import tf
from tf.transformations import euler_from_quaternion

import torch
from PIL import Image as PILImage
import mobileclip

from ultralytics import YOLO
import numpy as np
import cv2
import time

KNOWN_OBJECT_THRESHOLD = 0.4
UNKNOWN_OBJECT_THRESHOLD = 0.25

IOU_THRESHOLD = 0.1  # Set the IoU threshold for NMS
COLORS = [(255,50,50), (207,49,225), (114,15,191), (22,0,222), (0,177,122), (34,236,169),
          (34,236,81), (203,203,47), (205,90,23), (102,68,16), (168,215,141), (185,167,215)]

GEMINI_COLORS = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'yellow']
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
        """
        Determine if an object is overlapped with the map.
        Args:
            x (float): The x-coordinate of the object's center in meters.
            y (float): The y-coordinate of the object's center in meters.
            width (float): The width of the object in meters.
            height (float): The height of the object in meters.
        Returns:
            bool: True if the object is overlapped with the map, False otherwise.
        Notes:
            - The function calculates the object's radius as half of the minimum of its width and height.
            - It checks if the object's center is within the map boundaries.
            - It calculates the grid indices of the object's bounding box.
            - It counts the number of occupied cells within the bounding box.
            - It returns True if more than 20% of the cells within the bounding box are occupied.
        """
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

        return occupied_percentage > 50  # Return True if more than 50% of the cells are occupied


class Detector:
    
    def __init__(self):
        # Load the YOLO detector model
        weights_file = rospy.get_param('~weights_file', '../weights/osod.pt')
        self.model = YOLO(weights_file)

        # Get the corresponding labels for the classes
        labels_file = rospy.get_param('~labels_file', 'labels.txt')
        with open(labels_file, 'r') as f:
            self.labels = f.read().splitlines()
        print("Using model " + weights_file)

        self.model_conf = min(KNOWN_OBJECT_THRESHOLD, UNKNOWN_OBJECT_THRESHOLD)

        # Load the CLIP model and tokenizer
        clip_model = rospy.get_param('~clip_model', 'checkpoints/mobileclip_s0.pt')
        root_dir = rospy.get_param('~root_dir', None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _, self.clip_preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=clip_model, root_dir=root_dir, device=self.device)
        self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0', root_dir=root_dir)

        # Initialize variables for storing object names and text features for the CLIP model
        self.object_names = ["cone"]
        self.text = None
        self.text_features = None

        self.tall = rospy.get_param('~tall', False)  # True if camera mounted on tall robot (i.e. upside down)

        # These keep track of whether the robot is turning or not, 
        # The depth images produced when turning are not reliable, so we need to ignore them
        self.is_turning = False
        self.time_since_turning = 0

        # Keep track if static so don't call LLM repeatedly
        self.is_static = False
        self.queried_while_static = False
        self.time_since_static = 0

        # Create the publisher that will show image with bounding boxes
        self.boxes_publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)

        # Create the publishers that send known objects and unknown objects
        self.detected_object_publisher = rospy.Publisher('/detected_objects', DetectedObjectArray, queue_size=10)
        self.unknown_object_publisher = rospy.Publisher('/unknown_objects', DetectedObjectWithImageArray, queue_size=10)

        # Get the camera parameters to compute depth correctly
        camera_info_msg = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        camera_info = camera_info_msg.K
        camera_info = np.array(camera_info).reshape(3, 3)
        self.fx = camera_info[0, 0]
        self.fy = camera_info[1, 1]
        self.cx = camera_info[0, 2]
        self.cy = camera_info[1, 2]

        # Get the map and create a StochOccupancyGrid2D object
        # self.map keeps track of the original map plus objects that have been detected
        # so that we don't need to query the LLM multiple times for the same object
        self.map_msg = rospy.wait_for_message("/map", OccupancyGrid)
        self.map = StochOccupancyGrid2D(self.map_msg.info.resolution, 
                                             self.map_msg.info.width, 
                                            self.map_msg.info.height,
                                 self.map_msg.info.origin.position.x,
                                 self.map_msg.info.origin.position.y,
                                                                   5,
                                                   self.map_msg.data)

        # Create the transform listener to get current location of mobile robot
        self.tf_listener = tf.TransformListener()
        
        # Subscribe to  the cmd_vel topic to determine if the robot is turning or not
        self.cmd_vel_subscriber = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_callback, queue_size=1)    

        # Subscribe to the RGB and depth images, and create a time synchronizer
        self.rbg_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rbg_sub, self.depth_sub], 1, 0.1)
        self.ts.registerCallback(self.unifiedCallback)

        # Subscribe to the labeled unknown objects
        self.labeled_sub = rospy.Subscriber("/labeled_unknown_objects", LabeledObjectArray, self.labeled_callback, queue_size=3)

    def unifiedCallback(self, rgb_data, depth_data):
        """
        Callback function to process synchronized RGB and depth data for object detection.
        Args:
            rgb_data (sensor_msgs.msg.Image): The RGB image data.
            depth_data (sensor_msgs.msg.Image): The depth image data.
        Returns:
            None
        This function performs the following steps:
        1. Checks if the robot is turning and returns early if it is.
        2. Retrieves the transform from the map frame to the camera_link frame.
        3. Converts the RGB image data to a numpy array.
        4. Rotates the image if the camera is mounted on a tall robot.
        5. Performs object detection using a YOLO model.
        6. Extracts bounding boxes, confidence scores, and class labels from the detection results.
        7. Draws bounding boxes and labels on the image for detected objects.
        8. Creates and populates DetectedObjectArray and DetectedObjectWithImageArray messages for known and unknown objects, respectively.
        9. Converts the depth image data to a numpy array and rotates it if necessary.
        10. Filters detected objects based on depth information and map overlap.
        11. Publishes the image with all bounding boxes.
        12. Publishes the known and unknown detected objects.
        Note:
            - The function uses ROS (Robot Operating System) for message passing and transformations.
            - The function assumes that the YOLO model, ROS publishers, and other necessary components are properly initialized.
        """
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

        if self.tall:
            # Rotate image 180 degrees if mounted on tall robot
            image = cv2.rotate(image, cv2.ROTATE_180)

        # Perform object detection using YOLO
        results = self.model.predict(image, device=0, conf=self.model_conf, agnostic_nms=True, iou=IOU_THRESHOLD, verbose=False)

        detect_time = time.time()
        
        # Get the bounding boxes, confidence, and class labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        num_detected = len(clss)
        num_known_detected = np.sum((conf > KNOWN_OBJECT_THRESHOLD) & (clss != 0))
        num_unknown_detected = np.sum((conf > UNKNOWN_OBJECT_THRESHOLD) & (clss == 0))

        image_with_boxes = image.copy()  # Image to show all bounding boxes
        image_with_unknown_boxes = image.copy()  # Image to show only unknown bounding boxes
        clip_names = {}
        for i in range(num_detected):

            # Make sure the confidence is above the threshold
            if clss[i] == 0 and conf[i] < UNKNOWN_OBJECT_THRESHOLD:
                continue
            elif clss[i] != 0 and conf[i] < KNOWN_OBJECT_THRESHOLD:
                continue

            # Draw box and label on the image
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), COLORS[clss[i]], 2)

            if clss[i] != 0:
                cv2.putText(image_with_boxes, f"{self.labels[clss[i]]} {conf[i]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)
            else:
                clip_name = self.clip_classify(image[y1:y2, x1:x2])
                if clip_name != 'unknown':
                    # If the object is known, draw the label on the image
                    cv2.putText(image_with_boxes, f"CLIP: {clip_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)
                    # Change clss[i] to not be 0 so that it is considered a known object
                    clss[i] = -1
                    clip_names[i] = clip_name
                else: 
                    # CLIP didn't classify the object, so it is unknown
                    cv2.putText(image_with_boxes, f"{self.labels[clss[i]]} {conf[i]:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)

            # Note: we don't draw on the image_with_unknown_boxes because we are going to filter them first, only objects which don't intersect with 
            # the map or objects we haven't seen before will be added.

        # Create the DetectedObjectArray message for storing known detected objects
        detection_array = DetectedObjectArray()
        detection_array.header.stamp = rospy.Time.now()
        detection_array.header.frame_id = 'map'
        detection_array.objects = []

        # Create the DetectedObjectImageArray message for storing unknown detected objects
        unknown_object_array = DetectedObjectWithImageArray()
        unknown_object_array.header.stamp = rospy.Time.now()
        unknown_object_array.header.frame_id = 'map'
        unknown_object_array.objects = []

        data_dict = {}
        data_count = 0
        if num_known_detected > 0 or num_unknown_detected > 0:
            # Convert depth image to numpy array
            depth = np.frombuffer(depth_data.data, dtype=np.uint16).reshape(depth_data.height, depth_data.width)

            if self.tall:
                # Rotate the depth image 180 degrees
                depth = np.flip(depth)
        
            # Get only objects that are not overlapped with the map (use the depth + stored map to determine this)
            (data_dict, 
            data_count, 
            detection_array, 
            unknown_object_array,
            image_with_boxes,
            image_with_unknown_boxes) = self.filter_objects(boxes, conf, clss, num_detected, depth, trans, rot, 
                                            data_dict, data_count, detection_array, 
                                            unknown_object_array, image_with_boxes, image_with_unknown_boxes, image, clip_names)

        # Publish the image with all bounding boxes
        image_msg = Image()
        image_msg.data = image_with_boxes.tobytes()
        image_msg.height = image_with_boxes.shape[0]
        image_msg.width = image_with_boxes.shape[1]
        image_msg.encoding = 'rgb8'
        image_msg.step = 3 * image_with_boxes.shape[1]
        image_msg.header.stamp = rospy.Time.now()
        self.boxes_publisher.publish(image_msg)

        # Publish the known objects
        if len(detection_array.objects) > 0:
            self.detected_object_publisher.publish(detection_array)

        # Publish the unknown objects
        if len(unknown_object_array.objects) > 0:   
            # print("Have Non-Overlapping Unknown Objects")

            # Convert image to the ROS format
            _, buffer = cv2.imencode('.jpg', image_with_unknown_boxes)
            unknown_object_array.data = np.array(buffer).tobytes()

            # print(time.time() - self.time_since_static)
            if not self.is_static or not self.queried_while_static or time.time() - self.time_since_static > 10:
                # Only publish if not static
                self.unknown_object_publisher.publish(unknown_object_array)  # Publish the unknown objects

                print("Published unknown objects")

                if self.is_static:
                    self.queried_while_static = True
                    self.time_since_static = time.time()

        end_time = time.time()
        # print('Detection time: {}'.format(detect_time - start_time))
        # print('Elapsed time: {}'.format(end_time - start_time))

    def filter_objects(self, boxes, conf, clss, num_detected, depth, trans, rot, data_dict, data_count, detection_array, 
                           unknown_object_array, image_with_boxes, unknown_image, image, clip_names):

        # Get location of camera in map frame
        x_camera = trans[0]
        y_camera = trans[1]
        _, _, theta_camera = euler_from_quaternion(rot)
        theta_camera = theta_camera
        
        num_unknown = 0
        for i in range(num_detected):
            class_num = clss[i]
            if class_num == -1:
                class_name = clip_names[i]
            else:
                class_name = self.labels[class_num]

            
            class_score = conf[i]

            # Skip objects without a high enough confidence score
            if class_num <= 0 and class_score < UNKNOWN_OBJECT_THRESHOLD:
                continue
            elif class_num != 0 and class_score < KNOWN_OBJECT_THRESHOLD:
                continue
            
            # Get the bounding box coordinates  
            x_min, y_min, x_max, y_max = boxes[i]
            x1 = int(x_min)
            y1 = int(y_min)
            x2 = int(x_max)
            y2 = int(y_max)

            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            # Get the estimated depth of the object (in meters)
            estimated_depth = self.get_depth(x1, y1, x2, y2, depth)

            # Write the estimated depth on the image
            image_with_boxes = cv2.putText(image_with_boxes, 'Depth: {:.2f} m'.format(estimated_depth), (x1, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # Skip objects with invalid depth values (0) or too far away (> 5 meters) since they are unreliable
            if estimated_depth == 0 or estimated_depth > 5:
                new_object = DetectedObjectWithImage()
                new_object.class_name = class_name
                new_object.probability = class_score
                new_object.x1 = x1
                new_object.y1 = y1
                new_object.x2 = x2
                new_object.y2 = y2
                new_object.color = "none"
                unknown_object_array.objects.append(new_object)
                continue

            # Get the object's position in the camera frame using the estimated depth
            local_x_min = (x1 - self.cx) * estimated_depth / self.fx  # meters (left-most point)
            local_z_min = (y1 - self.cy) * estimated_depth / self.fy  # meters (bottom-most point)
            local_x_max = (x2 - self.cx) * estimated_depth / self.fx  # meters (right-most point)
            local_z_max = (y2 - self.cy) * estimated_depth / self.fy  # meters (top-most point)
            local_x_center = (x_center - self.cx) * estimated_depth / self.fx  # meters
            local_z_center = (y_center - self.cy) * estimated_depth / self.fy  # meters

            # Calculate the object's width and height
            object_width = local_x_max - local_x_min
            object_height = local_z_max - local_z_min
            
            if object_height < 0.1 or object_width < 0.1:
                continue  # Skip objects that are too small

            # Choose the object depth (i.e. width in the direction of the camera) as the minimum of width, height, and 0.33 meters
            object_depth = np.min([object_width, object_height, 0.33])
            
            # hypot is the distance from camera to object's 3D center
            hypot = np.sqrt(local_x_center**2 + (estimated_depth + object_depth/2)**2)

            # Calculate the object's position in the map frame
            theta_object = np.arctan2(local_x_center, estimated_depth)
            x_map = x_camera + (hypot + object_width/2) * np.cos(theta_camera - theta_object)  # Add object width/2 to get center of object (assume width=depth)
            y_map = y_camera + (hypot + object_width/2) * np.sin(theta_camera - theta_object)

            is_overlapped = False
            if self.map.is_overlapped(x_map, y_map, object_width, object_height):
                image_with_boxes = cv2.putText(image_with_boxes, 'Overlapped', (x1, y1+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                is_overlapped = True

            # Add the object to the data dictionary
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

            # Create DetectedObject message and add to the DetectedObjectArray message
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

            # Add unknown objects to the unknown_object_array if not overlapped with the map
            if not is_overlapped and class_name == 'unknown':

                # Get the part of the image within the bounding box
                unknown_object = DetectedObjectWithImage()
                unknown_object.class_name = class_name
                unknown_object.probability = class_score
                unknown_object.pose = detected_object_pose
                unknown_object.width = object_width
                unknown_object.x1 = x1
                unknown_object.y1 = y1
                unknown_object.x2 = x2
                unknown_object.y2 = y2
                # Limit the number of unknown objects to the number of colors available
                if num_unknown < len(GEMINI_COLORS):
                    unknown_object.color = GEMINI_COLORS[num_unknown]

                unknown_object.data = image[y1:y2, x1:x2].tobytes()
                unknown_object_array.objects.append(unknown_object)

                num_unknown += 1
                # unknown_image = cv2.rectangle(unknown_image, (x1, y1), (x2, y2), COLOR_CODES[len(unknown_object_array.objects)-1], 2)

            else:
                # Add the object to the unknown_object_array
                new_object = DetectedObjectWithImage()
                new_object.class_name = class_name
                new_object.probability = class_score
                new_object.pose = detected_object_pose
                new_object.width = object_width
                new_object.x1 = x1
                new_object.y1 = y1
                new_object.x2 = x2
                new_object.y2 = y2
                new_object.color = "none"
                unknown_object_array.objects.append(new_object)
            
        return data_dict, data_count, detection_array, unknown_object_array, image_with_boxes, unknown_image

    def get_depth(self, x_min, y_min, x_max, y_max, depth):
        """
        Calculate the estimated depth of a region in a depth image.
        This function extracts the depth values within a specified bounding box
        from a depth image, removes zero values, and returns the 25th percentile
        depth value converted to meters.
        Args:
            x_min (int): The minimum x-coordinate of the bounding box.
            y_min (int): The minimum y-coordinate of the bounding box.
            x_max (int): The maximum x-coordinate of the bounding box.
            y_max (int): The maximum y-coordinate of the bounding box.
            depth (np.ndarray): The depth image as a 2D numpy array.
        Returns:
            float: The estimated depth in meters. Returns 0.0 if no valid depth
            values are found within the bounding box.
        """
        # Get the depth values within the bounding box
        depth_values = depth[int(y_min):int(y_max), int(x_min):int(x_max)].flatten()

        # Remove zero depth values
        depth_values = depth_values[depth_values > 0]

        if len(depth_values) == 0:
            return 0.0  # 0 since no valid depth values are found

        # Get the 25th percentile depth
        estimated_depth = np.percentile(depth_values, 25) / 1000.0  # Convert to meters (/1000)

        return estimated_depth

    def labeled_callback(self, msg):
        # Update the object names and text features
        names_changed = False
        for obj in msg.objects:
            if obj.class_name not in self.object_names and len(self.object_names) < 10:
                self.object_names.append(obj.class_name)
                names_changed = True

            x_map = obj.pose.position.x
            y_map = obj.pose.position.y
            object_width = obj.width
            # self.map.add_to_map(x_map, y_map, object_width)

        if names_changed:
            self.update_text_features()


    def clip_classify(self, img):
        
        if len(self.object_names) < 2 or self.text_features is None:
            # We require at least 2 clip names to classify
            return "unknown"

        # Preprocess the image
        img = PILImage.fromarray(img)
        img = self.clip_preprocess(img).unsqueeze(0)

        # Encode the image
        with torch.no_grad(), torch.cuda.amp.autocast():
            img = img.to(self.device, dtype=torch.float16)
            img_features = self.clip_model.encode_image(img)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # Calculate the similarity scores between the image and text features
            text_scores = (100.0 * img_features @ self.text_features.T)
        
        ranked_scores = torch.argsort(text_scores, descending=True).cpu().numpy()[0]
        # Get ratio of top score to second score
        text_scores = text_scores.cpu().numpy()[0]
        ratio = text_scores[ranked_scores[0]] / text_scores[ranked_scores[1]]

        if ratio > 1.4:
            return self.object_names[ranked_scores[0]]
        else:
            return "unknown"

        
    def update_text_features(self):

        # Get updated tokens
        self.text = self.tokenizer(self.object_names).to(self.device)

        # Get the text features
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.text_features = self.clip_model.encode_text(self.text)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


    def cmd_vel_callback(self, msg):
        """
        Callback function for processing velocity command messages.
        This function is triggered when a new velocity command message is received.
        It updates the `is_turning` attribute based on the angular velocity of the message.
        If the absolute value of the angular velocity (`msg.angular.z`) is greater than 0.07,
        it sets `is_turning` to True and updates `time_since_turning` to the current time.
        If the time elapsed since the last turn is less than 0.25 seconds, it keeps `is_turning` as True.
        Otherwise, it sets `is_turning` to False.
        Args:
            msg (geometry_msgs.msg.Twist): The velocity command message containing linear and angular velocities.
        """
        if np.abs(msg.angular.z) > 0.15:
            self.is_turning = True
            self.time_since_turning = time.time()
        elif time.time() - self.time_since_turning < 0.25:
            self.is_turning = True
        else:
            self.is_turning = False

        if msg.linear.x == 0 and msg.angular.z == 0:
            if not self.is_static:
                self.time_since_static = time.time()
            
            self.is_static = True

        else:
            self.is_static = False
            self.queried_while_static = False


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    
    # Initialize the ROS node
    rospy.init_node('object_detection', anonymous=True)

    # Create the detector object and run it
    detector = Detector()
    detector.run()
    