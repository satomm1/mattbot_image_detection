#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import PoseWithCovariance, Pose, Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from image_detection_with_unknowns.msg import LabeledObject, LabeledObjectArray
import message_filters
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

import tf
from tf.transformations import euler_from_quaternion

# Requirements for Mobile Clip
import torch
from PIL import Image as PILImage
import mobileclip

from ultralytics import YOLO
import numpy as np
import cv2
import time

"""
This script performs object detection using a YOLO model on RGB and depth images from a camera.
It publishes images with bounding boxes and detected unknown objects.
Classes:
    Detector: A class that handles the detection of objects using YOLO and publishes results.
Functions:
    __init__(self): Initializes the Detector class, loads the YOLO model, and sets up publishers and subscribers.
    unifiedCallback(self, rgb_data, depth_data): Callback function that processes synchronized RGB and depth images, performs object detection, and publishes results.
    run(self): Spins the ROS node to keep it running.
Constants:
    KNOWN_OBJECT_THRESHOLD (float): Confidence threshold for known objects.
    UNKNOWN_OBJECT_THRESHOLD (float): Confidence threshold for unknown objects.
    IOU_THRESHOLD (float): Intersection over Union threshold for Non-Maximum Suppression.
    COLORS (list): List of colors for bounding boxes.
    GEMINI_COLORS (list): List of color names for unknown objects.
    COLOR_CODES (list): List of color codes for unknown objects.
Usage:
    Run this script as a ROS node to perform object detection on incoming camera images and publish results.
"""

KNOWN_OBJECT_THRESHOLD = 0.4
UNKNOWN_OBJECT_THRESHOLD = 0.25

IOU_THRESHOLD = 0.1  # Set the IoU threshold for NMS
COLORS = [(255,50,50), (207,49,225), (114,15,191), (22,0,222), (0,177,122), (34,236,169),
          (34,236,81), (203,203,47), (205,90,23), (102,68,16), (168,215,141), (185,167,215)]

GEMINI_COLORS = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'yellow']
COLOR_CODES = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128), (255, 192, 203), (255, 165, 0), (255, 255, 0)]

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

        # Load the CLIP model and tokenizer
        clip_model = rospy.get_param('~clip_model', 'checkpoints/mobileclip_s0.pt')
        root_dir = rospy.get_param('~root_dir', None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model, _, self.clip_preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained=clip_model, root_dir=root_dir, device=self.device)
        self.tokenizer = mobileclip.get_tokenizer('mobileclip_s0', root_dir=root_dir)

        # Initialize variables for storing object names and text features for the CLIP model
        self.object_names = []
        self.text = None
        self.text_features = None

        # Load from a parameter
        self.tall = rospy.get_param('~tall', False)  # True if camera mounted on tall robot (i.e. upside down)

        # Create the publisher that will show image with bounding boxes
        self.first_time = True
        self.boxes_publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)

        # Create the publisher that sends unknown objects
        self.unknown_pub = rospy.Publisher('/unknown_objects', DetectedObjectWithImageArray, queue_size=1)

        # Subscribe to the RGB and depth images, and create a time synchronizer
        self.rbg_sub = message_filters.Subscriber('/camera/color/image_raw', Image)
        self.depth_sub = message_filters.Subscriber('/camera/depth/image_raw', Image)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.rbg_sub, self.depth_sub], 1, 0.1)
        self.ts.registerCallback(self.unifiedCallback)
        # self.saved_image = False

        # Subscribe to the labeled unknown objects
        self.labeled_sub = rospy.Subscriber("/labeled_unknown_objects", LabeledObjectArray, self.labeled_callback, queue_size=3)

        print("Finished setup")

    def unifiedCallback(self, rgb_data, depth_data):

        start_time = time.time()

        # Get the image from the message
        image = np.frombuffer(rgb_data.data, dtype=np.uint8).reshape(rgb_data.height, rgb_data.width, -1)

        if self.tall:
            # Rotate image 180 degrees if mounted on tall robot
            image = cv2.rotate(image, cv2.ROTATE_180)

        # if not self.saved_image:
        #     # Save the first image as a reference
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     cv2.imwrite('/workspace/catkin_ws/src/image_detection_with_unknowns/scripts/reference_image.jpg', image)
        #     self.saved_image = True

        # Perform object detection using YOLO
        results = self.model.predict(image, device=0, conf=0.20, agnostic_nms=True, iou=IOU_THRESHOLD, verbose=False)

        detect_time = time.time()
        
        # Get the bounding boxes, confidence, and class labels
        boxes = results[0].boxes.xyxy.cpu().numpy()
        conf = results[0].boxes.conf.cpu().numpy()
        clss = results[0].boxes.cls.cpu().numpy().astype(int)
        num_detected = len(clss)

        # Create the DetectedObjectImageArray message for storing unknown objects
        unknown_object_array = DetectedObjectWithImageArray()
        unknown_object_array.header.stamp = rospy.Time.now()
        unknown_object_array.header.frame_id = 'map'
        unknown_object_array.objects = []

        image_with_boxes = image.copy()  # Image to show all bounding boxes
        image_with_unknown_boxes = image.copy()  # Image to show only unknown bounding boxes
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
                if y1 - 10 > 0:
                    cv2.putText(image_with_boxes, f"{self.labels[clss[i]]}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)
                else:
                    cv2.putText(image_with_boxes, f"{self.labels[clss[i]]}", (x1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)
                    # cv2.putText(image_with_boxes, f"{self.labels[clss[i]]} {conf[i]:.2f}", (x1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)

                new_object = DetectedObjectWithImage()
                new_object.class_name = self.labels[clss[i]]
                new_object.probability = conf[i]
                new_object.color = "none"
                new_object.data = image[y1:y2, x1:x2].tobytes()
                new_object.x1 = x1
                new_object.y1 = y1
                new_object.x2 = x2
                new_object.y2 = y2
                unknown_object_array.objects.append(new_object)

            # If the object is unknown, first compare to CLIP, then add it to the unknown_object_array and draw on unknown image if truly unknown
            if clss[i] == 0 and len(unknown_object_array.objects) < len(GEMINI_COLORS):
                
                clip_name = self.clip_classify(image[y1:y2, x1:x2])
                if clip_name != "unknown":
                    # If the object is known, draw the label on the image
                    cv2.putText(image_with_boxes, f"CLIP: {clip_name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)

                    new_object = DetectedObjectWithImage()
                    new_object.class_name = clip_name
                    new_object.probability = conf[i]
                    new_object.color = "none"
                    new_object.data = image[y1:y2, x1:x2].tobytes()
                    new_object.x1 = x1
                    new_object.y1 = y1
                    new_object.x2 = x2
                    new_object.y2 = y2
                    unknown_object_array.objects.append(new_object)
                else:
                    # CLIP didn't classify the object, so it is unknown
                    cv2.putText(image_with_boxes, f"{self.labels[clss[i]]}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLORS[clss[i]], 2)

                    unknown_object = DetectedObjectWithImage()
                    unknown_object.class_name = "unknown"
                    unknown_object.probability = conf[i]
                    unknown_object.color = GEMINI_COLORS[len(unknown_object_array.objects)]
                    unknown_object.data = image[y1:y2, x1:x2].tobytes()
                    unknown_object.x1 = x1
                    unknown_object.y1 = y1
                    unknown_object.x2 = x2
                    unknown_object.y2 = y2
                    # cv2.rectangle(image_with_unknown_boxes, (x1, y1), (x2, y2), COLOR_CODES[len(unknown_object_array.objects)], 2)
                    unknown_object_array.objects.append(unknown_object)

        # Publish the image with all bounding boxes
        image_msg = Image()
        image_msg.data = image_with_boxes.tobytes()
        image_msg.height = image_with_boxes.shape[0]
        image_msg.width = image_with_boxes.shape[1]
        image_msg.encoding = 'rgb8'
        image_msg.step = 3 * image_with_boxes.shape[1]
        image_msg.header.stamp = rospy.Time.now()
        self.boxes_publisher.publish(image_msg)

        # Publish the unknown objects
        if len(unknown_object_array.objects) > 0:
            unknown_object_array.header.stamp = rospy.Time.now()
            unknown_object_array.header.frame_id = 'map'

            # Convert image to the ROS format
            _, buffer = cv2.imencode('.jpg', image_with_unknown_boxes)
            unknown_object_array.data = np.array(buffer).tobytes()

            self.unknown_pub.publish(unknown_object_array)  # Publish the unknown objects

            if self.first_time:
                # Save image_with_boxes as labeled_image.jpg
                # convert image_with_boxes to RGB
                image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
                cv2.imwrite('/workspace/catkin_ws/src/image_detection_with_unknowns/scripts/labeled_image.jpg', image_with_boxes)
                self.first_time = False

        end_time = time.time()
        # print('Detection time: {}'.format(detect_time - start_time))
        # print('Elapsed time: {}'.format(end_time - start_time))


    def labeled_callback(self, msg):
        # Update the object names and text features
        names_changed = False
        for obj in msg.objects:
            if obj.class_name not in self.object_names:
                self.object_names.append(obj.class_name)
                names_changed = True

        if names_changed:
            self.update_text_features()

    
    def clip_classify(self, img):

        
        
        if len(self.object_names) < 2 or self.text_features is None:
            # We require at least 2 clip names to classify
            return "unknown"

        # Preprocess the image
        img = PILImage.fromarray(img)
        img = self.clip_preprocess(img).unsqueeze(0)

        # start_time = time.time()
        # Encode the image
        with torch.no_grad(), torch.cuda.amp.autocast():
            img = img.to(self.device, dtype=torch.float16)
            img_features = self.clip_model.encode_image(img)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            # end_time = time.time()
            # print("CLIP classification time: {}".format(end_time - start_time))

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

    def run(self):
        rospy.spin()

if __name__ == '__main__':

    # Initialize the ROS node
    rospy.init_node('object_detection', anonymous=True)

    # Create the detector object and run it
    detector = Detector()
    detector.run()
    # print("Load Success")