import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2, LaserScan
from std_msgs.msg import String
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from mattbot_image_detection.msg import LabeledObject, LabeledObjectArray
import message_filters
from nav_msgs.msg import OccupancyGrid, Odometry
from geometry_msgs.msg import Point
from mattbot_dds.msg import AgentLocation

import tf
from tf.transformations import euler_from_quaternion

import torch
from PIL import Image as PILImage
import mobileclip

from ultralytics import YOLO
import numpy as np
import cv2
import json
import math
import time

from detection_filters import (confident, drop_unknown_duplicates, heights_above_floor, line_of_sight_clear,
                               stands_off_floor)
from detection_timing import PairStats, TurningGate
from object_ranging import (OBJECT_WINDOW, PERSON_WINDOW, bbox_percentile_depth, box_width_height, effective_cx,
                            legacy_position, pixel_bearing, range_in_bearing_span, ray_point, scan_points_map,
                            surface_position, window_cluster_depth)

RANGE_METHODS = ("legacy", "robust")
PERSON_RANGE_SOURCES = ("camera", "lidar")

KNOWN_OBJECT_THRESHOLD = 0.6
UNKNOWN_OBJECT_THRESHOLD = 0.25

IOU_THRESHOLD = 0.1  # The original NMS IoU (class-agnostic); now ~nms_iou, default 0.5 class-aware

LABEL_FONT_SCALE = 0.75
LABEL_THICKNESS = 2
LABEL_HEIGHT = cv2.getTextSize("Ag", cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)[0][1]

def label_inside_box(y1):
    """True if a label drawn above the box at y1 would be cut off by the top of the image."""
    return y1 - 10 - LABEL_HEIGHT < 0

def draw_label(image, text, x1, y1, color):
    """Draw text above the box, or just inside its top edge when it would be cut off. Keeps it within the image width."""
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, LABEL_THICKNESS)[0][0]
    x = max(0, min(x1, image.shape[1] - text_width))
    y = y1 + LABEL_HEIGHT + 5 if label_inside_box(y1) else y1 - 10
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, color, LABEL_THICKNESS)
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

        # Which boxes are kept (detection_filters).
        #   nms_agnostic / nms_iou: class-aware NMS at IoU 0.5 by default. The original class-agnostic NMS at
        #     IoU 0.1 (nms_agnostic:=true nms_iou:=0.1) dropped any box overlapping a higher-scoring one by 10 %:
        #     people side by side became one, and people in front of a large object vanished.
        #   person_conf: people need this confidence (0.4; the other known classes keep 0.6, unknown 0.25), so
        #     distant and partly hidden people are found; the people tracker's confirmation absorbs the extra
        #     false positives. 0.6 restores the original.
        #   unknown_dedupe: drop an "unknown" box that duplicates a known-class box (IoU > unknown_dup_iou) or
        #     lies mostly inside a person box (unknown_inside_person of its area), so a bag or an arm is not
        #     reported as a static object. Needed with class-aware NMS, where classes no longer suppress each other.
        self.nms_agnostic = bool(rospy.get_param('~nms_agnostic', False))
        self.nms_iou = float(rospy.get_param('~nms_iou', 0.5))
        self.person_conf = float(rospy.get_param('~person_conf', 0.4))
        self.unknown_dedupe = bool(rospy.get_param('~unknown_dedupe', True))
        self.unknown_dup_iou = float(rospy.get_param('~unknown_dup_iou', 0.5))
        self.unknown_inside_person = float(rospy.get_param('~unknown_inside_person', 0.8))
        # Phantom static objects (detection_filters): non-person detections must
        #   floor_check: rise floor_min_height above the floor on at least floor_min_frac of their pixels near
        #     the object's depth (back-projected through the camera's height from TF and ~camera_pitch_deg,
        #     positive = looking down), so rugs, floor patterns, shadows and floor reflections are dropped;
        #   los_check: be visible: the ray from the camera to the object's near surface must not cross a wall of
        #     the static map (los_margin m short of it), so reflections in glass are dropped. People are exempt
        #     unless los_check_people (a door the map shows closed may be open).
        self.floor_check = bool(rospy.get_param('~floor_check', True))
        self.floor_min_height = float(rospy.get_param('~floor_min_height', 0.15))
        self.floor_min_frac = float(rospy.get_param('~floor_min_frac', 0.10))
        self.camera_pitch = math.radians(float(rospy.get_param('~camera_pitch_deg', 0.0)))
        self.los_check = bool(rospy.get_param('~los_check', True))
        self.los_check_people = bool(rospy.get_param('~los_check_people', False))
        self.los_margin = float(rospy.get_param('~los_margin', 0.15))
        self.camera_height = None  # base_footprint -> camera_link z, looked up once (a fixed joint)
        self.n_dropped_floor = self.n_dropped_los = 0
        self.person_classes = [i for i, name in enumerate(self.labels) if name == 'person']
        self.class_conf = {c: self.person_conf for c in self.person_classes}
        self.model_conf = min(KNOWN_OBJECT_THRESHOLD, UNKNOWN_OBJECT_THRESHOLD, self.person_conf)
        rospy.loginfo("Detections: NMS %s IoU %.2f; confidence person %.2f, known %.2f, unknown %.2f; unknown dedupe %s; "
                      "floor check %s (>= %.2f m on %.0f %% of pixels, pitch %.1f deg); line-of-sight check %s",
                      "class-agnostic" if self.nms_agnostic else "class-aware", self.nms_iou, self.person_conf,
                      KNOWN_OBJECT_THRESHOLD, UNKNOWN_OBJECT_THRESHOLD, "on" if self.unknown_dedupe else "off",
                      "on" if self.floor_check else "off", self.floor_min_height, 100 * self.floor_min_frac,
                      math.degrees(self.camera_pitch), "on" if self.los_check else "off")

        # # Load the CLIP model and tokenizer
        # clip_model = rospy.get_param('~clip_model', 'checkpoints/mobileclip_s0.pt')
        # root_dir = rospy.get_param('~root_dir', None)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.clip_model, _, self.clip_preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=clip_model, root_dir=root_dir, device=self.device)
        # self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0', root_dir=root_dir)

        # Initialize variables for storing object names and text features for the CLIP model
        self.object_names = ["cone"]
        self.text = None
        self.text_features = None

        self.tall = rospy.get_param('~tall', False)  # True if camera mounted on tall robot (i.e. upside down)

        # Display options for /camera/color/image_with_boxes (do not affect the published detections)
        self.display_min_conf = rospy.get_param('~display_min_conf', 0.0)  # Only draw objects with at least this confidence
        self.display_depth = rospy.get_param('~display_depth', True)  # Write the estimated depth on each box
        self.display_overlapped = rospy.get_param('~display_overlapped', False)  # Write "Overlapped" on boxes that overlap the map

        # How detections are placed (object_ranging). The defaults reproduce the original estimate exactly.
        #   range_method: legacy = 25th percentile of the box's depth, centre ~0.4 m behind a person's surface;
        #                 robust = nearest substantial surface in a central (torso) window, centre one body
        #                 half-depth behind it.
        #   person_range_source: camera, or lidar = people ranged by the /scan cluster inside the box's
        #                 bearing span (camera estimate as the fallback; works < 0.6 m and beyond the camera's range).
        self.range_method = rospy.get_param('~range_method', 'legacy')
        self.person_range_source = rospy.get_param('~person_range_source', 'camera')
        if self.range_method not in RANGE_METHODS:
            rospy.logwarn("Unknown ~range_method %r, using legacy (choices: %s)", self.range_method, RANGE_METHODS)
            self.range_method = 'legacy'
        if self.person_range_source not in PERSON_RANGE_SOURCES:
            rospy.logwarn("Unknown ~person_range_source %r, using camera (choices: %s)", self.person_range_source,
                          PERSON_RANGE_SOURCES)
            self.person_range_source = 'camera'
        # Camera-ranged detections farther than this (forward depth, m) are dropped. 8 m: the range the learned
        # planner's camera was trained on (the Astra's rated maximum); the original limit was 5 m. Beyond about 5 m
        # the depth is noisier (people are ranged by the LiDAR when person_range_source is lidar).
        self.max_camera_depth = float(rospy.get_param('~max_camera_depth', 8.0))
        self.person_centre_offset = rospy.get_param('~person_centre_offset', 0.15)  # m behind the torso surface
        self.lidar_centre_offset = rospy.get_param('~lidar_centre_offset', 0.10)    # m behind the scan returns
        self.lidar_max_range = rospy.get_param('~lidar_max_range', 8.0)             # m (people ranged by LiDAR)
        self.lidar_max_scan_age = rospy.get_param('~lidar_max_scan_age', 0.3)       # s between scan and image
        self.lidar_margin = rospy.get_param('~lidar_margin', 0.15)                  # box span shrink per side
        self.publish_range_debug = rospy.get_param('~publish_range_debug', False)
        self.latest_scan = None  # (stamp, frame_id, ranges, angles, range_min, range_max), swapped atomically
        self.current_image_stamp = rospy.Time(0)
        if self.person_range_source == 'lidar':
            rospy.Subscriber(rospy.get_param('~scan_topic', '/scan'), LaserScan, self.scan_callback, queue_size=1)
        self.range_debug_publisher = (rospy.Publisher('/detected_objects_range_debug', String, queue_size=5)
                                      if self.publish_range_debug else None)
        rospy.loginfo("Object ranging: range_method=%s person_range_source=%s%s", self.range_method,
                      self.person_range_source, " (debug on)" if self.publish_range_debug else "")

        # Skip frames taken while the robot turns fast: the colour and depth images of a pair are taken at
        # slightly different times, so a fast turn shifts one against the other (detection_timing.TurningGate).
        #   turning_source: odom (default) = the MEASURED yaw rate at the image time; cmd = the original gate on
        #   the commanded /cmd_vel rate (it fires on every small steering correction, and its state freezes
        #   while /cmd_vel is silent).
        self.turning_source = rospy.get_param('~turning_source', 'odom')
        if self.turning_source not in ('odom', 'cmd'):
            rospy.logwarn("Unknown ~turning_source %r, using odom (choices: odom, cmd)", self.turning_source)
            self.turning_source = 'odom'
        self.turning_threshold = rospy.get_param('~turning_threshold', 0.8)  # |yaw rate| (rad/s) above which the robot is turning
        self.turning_cooldown = rospy.get_param('~turning_cooldown', 0.15)  # Keep ignoring frames this long after a turn (s)
        self.is_turning = False
        self.time_since_turning = 0
        self.turning_gate = TurningGate(self.turning_threshold, self.turning_cooldown)
        self.n_skipped_turning = 0
        if self.turning_source == 'odom':
            rospy.Subscriber(rospy.get_param('~odom_topic', '/odom'), Odometry, self.odom_callback, queue_size=20)

        # Keep track if static so don't call LLM repeatedly
        self.is_static = False
        self.queried_while_static = False
        self.time_since_static = 0
        self.last_query_time = 0

        # Create the publisher that will show image with bounding boxes
        self.boxes_publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)
        self.annotate = False  # this frame draws the annotated image (someone subscribes to it)

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
        # The static map's walls, for the line-of-sight check (occupancy >= 50; unknown cells are not walls)
        self.static_occupied = (np.asarray(self.map_msg.data, dtype=np.int16).reshape(
            self.map_msg.info.height, self.map_msg.info.width) >= 50)
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

        # Subscribe to the RGB and depth images, and create a time synchronizer.
        # queue_size 1 and a large buff_size: inference is slower than the camera's 30 fps, so with rospy's
        # defaults (unbounded queue, 64 kB buffer for 0.9 MB frames) every frame waits behind the backlog and
        # detections are about half a second old. With them, each callback gets the newest frame.
        image_queue = int(rospy.get_param('~image_queue_size', 1))
        image_buff = int(rospy.get_param('~image_buff_size', 2 ** 24))
        sub_kw = dict(queue_size=image_queue if image_queue > 0 else None, buff_size=image_buff)
        # sync_slop: the most the colour and depth stamps of a pair may differ (s). One frame at 30 fps: a
        # looser pairing shifts the depth image against the colour boxes whenever anything moves.
        self.sync_slop = float(rospy.get_param('~sync_slop', 0.033))
        self.pair_stats = PairStats()
        self.rbg_sub = message_filters.Subscriber('/camera/color/image_raw', Image, **sub_kw)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image, **sub_kw)
        self.rbg_sub.registerCallback(self._count_rgb)
        self.depth_sub.registerCallback(self._count_depth)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rbg_sub, self.depth_sub], 1, self.sync_slop)
        self.ts.registerCallback(self.unifiedCallback)
        self._pairing_counts = (0, 0, 0)
        rospy.Timer(rospy.Duration(5.0), self._check_pairing)
        rospy.loginfo("Image input: queue_size=%s buff_size=%d sync_slop=%.3f s; turning gate: %s > %.2f rad/s",
                      sub_kw['queue_size'], image_buff, self.sync_slop, self.turning_source, self.turning_threshold)

        # Subscribe to other agent locations
        self.agent_location_subscriber = rospy.Subscriber('/agent_location', AgentLocation, self.agent_location_callback, queue_size=10)
        self.other_agent_locs = dict()

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
        # The time the image was taken: detections are stamped with it, and the camera pose is looked up
        # at it, so a tracker downstream (people_tracker_node) can estimate velocities and compensate for
        # the inference latency. Falls back to the latest transform (the old behaviour) if TF lacks that time.
        image_stamp = rgb_data.header.stamp if rgb_data.header.stamp.to_sec() > 0.0 else rospy.Time.now()
        self.pair_stats.add_pair(rgb_data.header.stamp.to_sec(), depth_data.header.stamp.to_sec())

        turning = (self.turning_gate.turning(image_stamp.to_sec()) if self.turning_source == 'odom'
                   else self.is_turning)
        if turning:
            # Can't rely on the colour/depth alignment while the robot turns fast
            self.n_skipped_turning += 1
            return

        start_time = time.time()
        self.current_image_stamp = image_stamp

        # Get the transform from map to camera_link
        try:
            try:
                self.tf_listener.waitForTransform("/map", "/camera_link", image_stamp, rospy.Duration(0.05))
                (trans, rot) = self.tf_listener.lookupTransform("/map", "/camera_link", image_stamp)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                (trans, rot) = self.tf_listener.lookupTransform("/map", "/camera_link", rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        # Get the image from the message
        image = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)

        if self.tall:
            # Rotate image 180 degrees if mounted on tall robot
            image = cv2.rotate(image, cv2.ROTATE_180)

        # Perform object detection using YOLO
        results = self.model.predict(image, device=0, conf=self.model_conf, agnostic_nms=self.nms_agnostic,
                                     iou=self.nms_iou, verbose=False)

        detect_time = time.time()
        
        # Get the bounding boxes, confidence, and class labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        num_detected = len(clss)
        # The boxes kept: per-class confidence, then open-set duplicates removed (detection_filters)
        keep = confident(conf, clss, unknown_class=0, unknown_conf=UNKNOWN_OBJECT_THRESHOLD,
                         known_conf=KNOWN_OBJECT_THRESHOLD, class_conf=self.class_conf)
        if self.unknown_dedupe:
            keep = drop_unknown_duplicates(boxes, clss, keep, unknown_class=0, person_classes=self.person_classes,
                                           dup_iou=self.unknown_dup_iou, inside_frac=self.unknown_inside_person)

        # Draw and publish the annotated image only when someone subscribes to it (rviz, image_view): copying,
        # drawing and sending a 0.9 MB frame on every detection is otherwise wasted. Nothing is drawn on the image
        # sent with unknown objects, so it is the frame itself.
        self.annotate = self.boxes_publisher.get_num_connections() > 0
        image_with_boxes = image.copy() if self.annotate else image  # Image to show all bounding boxes
        image_with_unknown_boxes = image  # Image sent with the unknown objects (not drawn on)
        clip_names = {}
        for i in range(num_detected):

            # Make sure the box is kept (confidence above its class's threshold, not an open-set duplicate)
            if not keep[i]:
                continue

            # Draw box and label on the image
            box = boxes[i]
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            draw = self.annotate and conf[i] >= self.display_min_conf
            if draw:
                cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), COLORS[clss[i]], 2)

            if clss[i] != 0:
                if draw:
                    draw_label(image_with_boxes, f"{self.labels[clss[i]]} {conf[i]:.2f}", x1, y1, COLORS[clss[i]])
            else:
                clip_name = self.clip_classify(image[y1:y2, x1:x2])
                if clip_name != 'unknown':
                    # If the object is known, draw the label on the image
                    if draw:
                        draw_label(image_with_boxes, f"CLIP: {clip_name}", x1, y1, COLORS[clss[i]])
                    # Change clss[i] to not be 0 so that it is considered a known object
                    clss[i] = -1
                    clip_names[i] = clip_name
                elif draw:
                    # CLIP didn't classify the object, so it is unknown
                    draw_label(image_with_boxes, f"{self.labels[clss[i]]} {conf[i]:.2f}", x1, y1, COLORS[clss[i]])

            # Note: we don't draw on the image_with_unknown_boxes because we are going to filter them first, only objects which don't intersect with 
            # the map or objects we haven't seen before will be added.

        # Create the DetectedObjectArray message for storing known detected objects
        detection_array = DetectedObjectArray()
        detection_array.header.stamp = image_stamp  # image time, not after inference
        detection_array.header.frame_id = 'map'
        detection_array.objects = []

        # Create the DetectedObjectImageArray message for storing unknown detected objects
        unknown_object_array = DetectedObjectWithImageArray()
        unknown_object_array.header.stamp = rospy.Time.now()
        unknown_object_array.header.frame_id = 'map'
        unknown_object_array.objects = []

        data_dict = {}
        data_count = 0
        if keep.any():
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
                                            unknown_object_array, image_with_boxes, image_with_unknown_boxes, image, clip_names,
                                            keep)

        # Publish the image with all bounding boxes (only drawn when someone subscribes)
        if self.annotate:
            image_msg = Image()
            image_msg.data = image_with_boxes.tobytes()
            image_msg.height = image_with_boxes.shape[0]
            image_msg.width = image_with_boxes.shape[1]
            image_msg.encoding = 'rgb8'
            image_msg.step = 3 * image_with_boxes.shape[1]
            image_msg.header.stamp = rospy.Time.now()
            self.boxes_publisher.publish(image_msg)

        # Publish the known objects
        self.detected_object_publisher.publish(detection_array)

        # Publish the unknown objects
        if len(unknown_object_array.objects) > 0:   
            # print("Have Non-Overlapping Unknown Objects")

            # Convert image to the ROS format
            _, buffer = cv2.imencode('.jpg', image_with_unknown_boxes)
            unknown_object_array.data = np.array(buffer).tobytes()

            # print(time.time() - self.time_since_static)
            time_since_last_query = time.time() - self.last_query_time
            if (not self.is_static and time_since_last_query > 1) or not self.queried_while_static or time.time() - self.time_since_static > 20:
                # Only publish if not static
                self.unknown_object_publisher.publish(unknown_object_array)  # Publish the unknown objects

                # print("Published unknown objects")
                self.last_query_time = time.time()

                if self.is_static:
                    self.queried_while_static = True
                    self.time_since_static = time.time()

        end_time = time.time()
        # print('Detection time: {}'.format(detect_time - start_time))
        # print('Elapsed time: {}'.format(end_time - start_time))

    def filter_objects(self, boxes, conf, clss, num_detected, depth, trans, rot, data_dict, data_count, detection_array, 
                           unknown_object_array, image_with_boxes, unknown_image, image, clip_names, keep=None):

        # Get location of camera in map frame
        x_camera = trans[0]
        y_camera = trans[1]
        _, _, theta_camera = euler_from_quaternion(rot)
        theta_camera = theta_camera

        # Ranging inputs for the non-legacy options: the principal point in the (possibly rotated) image
        # the boxes are in, and this frame's LiDAR returns in the map frame
        cx_eff = effective_cx(self.cx, depth.shape[1], self.tall)
        scan_pts = self._scan_points() if self.person_range_source == 'lidar' else None
        range_debug = []
        
        num_unknown = 0
        for i in range(num_detected):
            class_num = clss[i]
            if class_num == -1:
                class_name = clip_names[i]
            else:
                class_name = self.labels[class_num]

            
            class_score = conf[i]

            # Skip boxes not kept (confidence below its class's threshold, or an open-set duplicate)
            if keep is not None:
                if not keep[i]:
                    continue
            elif class_num <= 0 and class_score < UNKNOWN_OBJECT_THRESHOLD:
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
            is_person = class_name == 'person'
            if self.range_method == 'robust':
                estimated_depth = window_cluster_depth(depth, x1, y1, x2, y2,
                                                       window=PERSON_WINDOW if is_person else OBJECT_WINDOW)
            else:
                estimated_depth = self.get_depth(x1, y1, x2, y2, depth)

            # People ranged by the LiDAR (its cluster, consistent with the camera depth when there is one)
            lidar_hit = None
            if is_person and self.person_range_source == 'lidar' and scan_pts is not None:
                camera_ok = 0 < estimated_depth <= self.max_camera_depth
                lidar_hit = range_in_bearing_span(
                    scan_pts, x_camera, y_camera, theta_camera,
                    pixel_bearing(x2, cx_eff, self.fx), pixel_bearing(x1, cx_eff, self.fx),
                    margin=self.lidar_margin, max_range=self.lidar_max_range,
                    prefer_range=self._camera_range(x_center, estimated_depth, cx_eff) if camera_ok else None)

            # Write the estimated depth on the image
            draw = self.annotate and class_score >= self.display_min_conf
            text_offset = LABEL_HEIGHT + 10 if label_inside_box(y1) else 0  # Move below the label if it is inside the box
            if draw and self.display_depth:
                depth_label = 'Depth: {:.2f} m'.format(estimated_depth)
                if lidar_hit is not None:
                    depth_label += '  L {:.2f} m'.format(lidar_hit[2])  # the LiDAR range used for this person
                image_with_boxes = cv2.putText(image_with_boxes, depth_label, (x1, y1+15+text_offset),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            if lidar_hit is not None:
                # The LiDAR range replaces the camera depth (and its validity window) for this person
                lidar_range = lidar_hit[2]
                lidar_bearing = pixel_bearing(x_center, cx_eff, self.fx)
                estimated_depth = lidar_range * math.cos(lidar_bearing)  # forward depth, for the width/height

            # Skip objects with invalid depth values (0) or too far away (> max_camera_depth) since they are unreliable
            if lidar_hit is None and (estimated_depth == 0 or estimated_depth > self.max_camera_depth):
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

            # The object's size at the depth used, and its position in the map frame
            if self.range_method == 'legacy' and lidar_hit is None:
                x_map, y_map, object_width, object_height = legacy_position(
                    x_camera, y_camera, theta_camera, x1, y1, x2, y2, estimated_depth, self.cx, self.cy, self.fx, self.fy)
                range_source = 'C'
            else:
                object_width, object_height = box_width_height(x1, y1, x2, y2, estimated_depth, self.fx, self.fy)
                if lidar_hit is not None:
                    x_map, y_map = ray_point(x_camera, y_camera, theta_camera, lidar_bearing,
                                             lidar_range + self.lidar_centre_offset)
                    range_source = 'L'
                else:
                    offset = self.person_centre_offset if is_person else 0.5 * min(object_width, object_height, 0.33)
                    x_map, y_map, _ = surface_position(x_camera, y_camera, theta_camera, x_center, estimated_depth,
                                                       cx_eff, self.fx, offset)
                    range_source = 'C'

            if object_height < 0.1 or object_width < 0.1:
                continue  # Skip objects that are too small

            # Phantom static objects: something flat on the floor, or seen through a wall (a reflection)
            phantom = self._phantom(is_person, depth, x1, y1, x2, y2, estimated_depth, x_camera, y_camera,
                                    theta_camera, x_map, y_map)
            if phantom == 'floor':
                self.n_dropped_floor += 1
                continue
            if phantom == 'los':
                self.n_dropped_los += 1
                continue

            if self.range_debug_publisher is not None:
                range_debug.append(self._range_debug(class_name, x1, y1, x2, y2, depth, x_camera, y_camera,
                                                     theta_camera, cx_eff, scan_pts, range_source, x_map, y_map))

            is_overlapped = False
            if self.map.is_overlapped(x_map, y_map, object_width, object_height):
                if draw and self.display_overlapped:
                    image_with_boxes = cv2.putText(image_with_boxes, 'Overlapped', (x1, y1+30+text_offset),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                is_overlapped = True

            # Now check if object might be the same as the robot:
            skip_object = False
            if self.other_agent_locs:
                for agent_id in list(self.other_agent_locs.keys()):
                    agent_x = self.other_agent_locs[agent_id][0]
                    agent_y = self.other_agent_locs[agent_id][1]

                    # Check if the object is within 1 meter of the agent's position
                    dist_to_agent = np.sqrt((x_map - agent_x)**2 + (y_map - agent_y)**2)
                    if dist_to_agent < 1:
                        # Skip this object since it is likely the robot
                        skip_object = True
            if skip_object:
                continue  # Skip this object since it is likely the robot

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
            
        if self.range_debug_publisher is not None and range_debug:
            self.range_debug_publisher.publish(String(data=json.dumps({
                "stamp": self.current_image_stamp.to_sec(), "range_method": self.range_method,
                "person_range_source": self.person_range_source, "camera": [x_camera, y_camera, theta_camera],
                "detections": range_debug})))

        return data_dict, data_count, detection_array, unknown_object_array, image_with_boxes, unknown_image

    # ------------------------------------------------------------------ phantom-object checks (detection_filters)

    def _camera_height_above_floor(self):
        """The optical centre's height above the floor (base_footprint -> camera_link z), once; None until TF has it."""
        if self.camera_height is None:
            try:
                (t, _) = self.tf_listener.lookupTransform('/base_footprint', '/camera_link', rospy.Time(0))
                self.camera_height = float(t[2])
                rospy.loginfo("Detector: camera %.3f m above the floor (floor check)", self.camera_height)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return None
        return self.camera_height

    def _phantom(self, is_person, depth, x1, y1, x2, y2, estimated_depth, x_camera, y_camera, theta_camera,
                 x_map, y_map):
        """'floor' or 'los' if a non-person detection fails a phantom check, else None (kept)."""
        if is_person and not self.los_check_people:
            return None
        if not is_person and self.floor_check and estimated_depth > 0:
            cam_h = self._camera_height_above_floor()
            if cam_h is not None:
                cy_eff = (depth.shape[0] - 1) - self.cy if self.tall else self.cy
                heights = heights_above_floor(depth, x1, y1, x2, y2, estimated_depth, self.fy, cy_eff, cam_h,
                                              self.camera_pitch)
                if stands_off_floor(heights, self.floor_min_height, self.floor_min_frac) is False:
                    return 'floor'
        if self.los_check and estimated_depth > 0:
            dx, dy = x_map - x_camera, y_map - y_camera
            dist = math.hypot(dx, dy)
            # The near surface along the ray to the object: its measured forward depth over the cosine of the
            # ray's angle off the optical axis (not centre minus half the width: a wide object's centre-to-edge
            # distance says nothing about how far its surface is), stopped los_margin short of it.
            cos_off = (dx * math.cos(theta_camera) + dy * math.sin(theta_camera)) / dist if dist > 0 else 1.0
            near = estimated_depth / max(cos_off, 0.2) - self.los_margin
            if dist > 0 and near > 0:
                end = (x_camera + dx * near / dist, y_camera + dy * near / dist)
                info = self.map_msg.info
                if not line_of_sight_clear(self.static_occupied, info.resolution,
                                           (info.origin.position.x, info.origin.position.y),
                                           (x_camera, y_camera), end):
                    return 'los'
        return None

    # ------------------------------------------------------------------ ranging helpers (object_ranging)

    def scan_callback(self, msg):
        """Keep the latest scan as one immutable tuple (atomic swap: the YOLO callback runs in another thread)."""
        n = len(msg.ranges)
        angles = msg.angle_min + msg.angle_increment * np.arange(n)
        self.latest_scan = (msg.header.stamp, msg.header.frame_id, np.asarray(msg.ranges, dtype=np.float64),
                            angles, msg.range_min, msg.range_max)

    def _scan_points(self):
        """This frame's LiDAR returns in the map frame (placed with the robot pose at SCAN time), or None
        when there is no scan within lidar_max_scan_age of the image."""
        scan = self.latest_scan
        if scan is None:
            return None
        stamp, frame, ranges, angles, rmin, rmax = scan
        if abs((self.current_image_stamp - stamp).to_sec()) > self.lidar_max_scan_age:
            return None
        frame = frame.lstrip('/') or 'laser_frame'
        try:
            try:
                (t, r) = self.tf_listener.lookupTransform('/map', frame, stamp)
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                (t, r) = self.tf_listener.lookupTransform('/map', frame, rospy.Time(0))
        except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return None
        return scan_points_map(ranges, angles, t[0], t[1], euler_from_quaternion(r)[2], rmin, rmax)

    def _camera_range(self, u_center, depth_m, cx_eff):
        """Range along the centre ray to the surface at forward depth ``depth_m``."""
        return depth_m / math.cos(pixel_bearing(u_center, cx_eff, self.fx))

    def _range_debug(self, class_name, x1, y1, x2, y2, depth, x_camera, y_camera, theta_camera, cx_eff, scan_pts,
                     used, x_map, y_map):
        """Every method's estimate side by side, for calibrating them against a tape measure."""
        is_person = class_name == 'person'
        u = (x1 + x2) / 2
        d_legacy = bbox_percentile_depth(depth, x1, y1, x2, y2)
        d_robust = window_cluster_depth(depth, x1, y1, x2, y2, window=PERSON_WINDOW if is_person else OBJECT_WINDOW)
        out = {"class": class_name, "box": [x1, y1, x2, y2], "used": used, "range_method": self.range_method,
               "position": [x_map, y_map]}
        if d_legacy > 0:
            lx, ly, _, _ = legacy_position(x_camera, y_camera, theta_camera, x1, y1, x2, y2, d_legacy,
                                           self.cx, self.cy, self.fx, self.fy)
            out["legacy"] = {"depth": d_legacy, "range": math.hypot(lx - x_camera, ly - y_camera), "xy": [lx, ly]}
        if d_robust > 0:
            w, h = box_width_height(x1, y1, x2, y2, d_robust, self.fx, self.fy)
            off = self.person_centre_offset if is_person else 0.5 * min(w, h, 0.33)
            rx, ry, rs = surface_position(x_camera, y_camera, theta_camera, u, d_robust, cx_eff, self.fx, off)
            out["robust"] = {"depth": d_robust, "surface_range": rs, "range": math.hypot(rx - x_camera, ry - y_camera),
                             "xy": [rx, ry]}
        if is_person and scan_pts is not None:
            hit = range_in_bearing_span(scan_pts, x_camera, y_camera, theta_camera,
                                        pixel_bearing(x2, cx_eff, self.fx), pixel_bearing(x1, cx_eff, self.fx),
                                        margin=self.lidar_margin, max_range=self.lidar_max_range,
                                        prefer_range=self._camera_range(u, d_robust, cx_eff) if d_robust > 0 else None)
            if hit is not None:
                out["lidar"] = {"surface_range": hit[2], "range": hit[2] + self.lidar_centre_offset,
                                "returns_xy": [hit[0], hit[1]]}
        return out

    def get_depth(self, x_min, y_min, x_max, y_max, depth):  # legacy; object_ranging.bbox_percentile_depth is a pinned copy
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
        
            return "unknown"

        # if len(self.object_names) < 2 or self.text_features is None:
        #     # We require at least 2 clip names to classify
        #     return "unknown"

        # # Preprocess the image
        # img = PILImage.fromarray(img)
        # img = self.clip_preprocess(img).unsqueeze(0)

        # # Encode the image
        # with torch.no_grad(), torch.cuda.amp.autocast():
        #     img = img.to(self.device, dtype=torch.float16)
        #     img_features = self.clip_model.encode_image(img)
        #     img_features /= img_features.norm(dim=-1, keepdim=True)

        #     # Calculate the similarity scores between the image and text features
        #     text_scores = (100.0 * img_features @ self.text_features.T)
        
        # ranked_scores = torch.argsort(text_scores, descending=True).cpu().numpy()[0]
        # # Get ratio of top score to second score
        # text_scores = text_scores.cpu().numpy()[0]
        # ratio = text_scores[ranked_scores[0]] / text_scores[ranked_scores[1]]

        # if ratio > 1.4:
        #     return self.object_names[ranked_scores[0]]
        # else:
        #     return "unknown"

        
    def update_text_features(self):

        pass

        # # Get updated tokens
        # self.text = self.tokenizer(self.object_names).to(self.device)

        # # Get the text features
        # with torch.no_grad(), torch.cuda.amp.autocast():
        #     self.text_features = self.clip_model.encode_text(self.text)
        #     self.text_features /= self.text_features.norm(dim=-1, keepdim=True)


    def odom_callback(self, msg):
        """Measured yaw rate for the turning gate (turning_source: odom)."""
        stamp = msg.header.stamp.to_sec() if msg.header.stamp.to_sec() > 0.0 else rospy.Time.now().to_sec()
        self.turning_gate.add_odom(stamp, msg.twist.twist.angular.z)

    def _count_rgb(self, _msg):
        self.pair_stats.n_rgb += 1

    def _count_depth(self, _msg):
        self.pair_stats.n_depth += 1

    def _check_pairing(self, _event):
        """Every 5 s: warn if colour and depth both arrive but never pair (sync_slop tighter than the
        camera's stamp offset); log the pairing gaps and the frames skipped for turning once a minute."""
        s = self.pair_stats
        rgb, dep, pairs = s.n_rgb - self._pairing_counts[0], s.n_depth - self._pairing_counts[1], \
            s.n_pairs - self._pairing_counts[2]
        self._pairing_counts = (s.n_rgb, s.n_depth, s.n_pairs)
        if rgb > 0 and dep > 0 and pairs == 0:
            rospy.logwarn("Detector: %d colour and %d depth images in 5 s but no pairs within sync_slop %.3f s; "
                          "the camera's colour and depth stamps are further apart: raise ~sync_slop",
                          rgb, dep, self.sync_slop)
        g = s.summary()
        if g is not None:
            rospy.loginfo_throttle(60.0, "Detector: colour/depth pair gap median %.1f ms, max %.1f ms; "
                                   "%d frames skipped while turning; phantom objects dropped: %d flat on the floor, "
                                   "%d behind a wall" % (1e3 * g[0], 1e3 * g[1], self.n_skipped_turning,
                                                         self.n_dropped_floor, self.n_dropped_los))

    def cmd_vel_callback(self, msg):
        """
        Callback function for processing velocity command messages.
        This function is triggered when a new velocity command message is received.
        It updates the `is_turning` attribute based on the angular velocity of the message.
        If the absolute value of the angular velocity (`msg.angular.z`) is greater than `turning_threshold`,
        it sets `is_turning` to True and updates `time_since_turning` to the current time.
        If the time elapsed since the last turn is less than `turning_cooldown` seconds, it keeps `is_turning` as True.
        Otherwise, it sets `is_turning` to False.
        Args:
            msg (geometry_msgs.msg.Twist): The velocity command message containing linear and angular velocities.
        """
        if np.abs(msg.angular.z) > self.turning_threshold:
            self.is_turning = True
            self.time_since_turning = time.time()
        elif time.time() - self.time_since_turning < self.turning_cooldown:
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

    def agent_location_callback(self, msg):
        """
        Callback function for processing agent location messages.
        This function is triggered when a new agent location message is received.
        It updates the `other_agent_locs` dictionary with the agent's ID and its corresponding position.
        Args:
            msg (mattbot_dds.msg.AgentLocation): The agent location message containing the agent's ID and position.
        """
        self.other_agent_locs[msg.agentID.data] = (msg.pose.position.x, msg.pose.position.y)


    def run(self):
        rospy.spin()


if __name__ == '__main__':
    
    # Initialize the ROS node
    rospy.init_node('object_detection', anonymous=True)

    # Create the detector object and run it
    detector = Detector()
    detector.run()
    