import rospy
from sensor_msgs.msg import Image
import cv2

from ultralytics import YOLO
import numpy as np
import time

import torch
import torchvision
from torchvision.models.detection import FasterRCNN, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.transforms import functional as F
from torchvision.ops import nms

"""
This is a test script for comparing the performance of the YOLOv8 model with the FasterRCNN model.
Primarily, this is used to see what sort of bounding boxes are detected when very low confidence score
thresholds are used.
"""

MODEL_TO_USE = "YOLO"
# MODEL_TO_USE = "FasterRCNN"

IOU_THRESHOLD = 0.4  # Set the IoU threshold for NMS


def callback(data):
    # Get the image from the message
    image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)
    image = cv2.rotate(image, cv2.ROTATE_180)

    if MODEL_TO_USE == "FasterRCNN":
        # Convert the image to a PyTorch tensor
        image_tensor = F.to_tensor(image)

        # Add a batch dimension to the tensor
        image_tensor = image_tensor.unsqueeze(0)

        # Run the image through the model
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            t1 = time.time()
            predictions = faster_model(image_tensor)
            t2 = time.time()
            print(f"Time taken: {t2-t1:.4f}s")

        boxes = predictions[0]['boxes']
        scores = predictions[0]['scores']
        labels = predictions[0]['labels']

        keep = nms(boxes, scores, IOU_THRESHOLD)
        boxes = boxes[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()

        image = image.copy()
        for box, label, score in zip(boxes, labels, scores):
            score_threshold = rospy.get_param('score_threshold', 0.2)
            if score > score_threshold and label-1 != 0:
                box = [int(i) for i in box]
                cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (43, 204, 10), 2)
                label_name = coco_labels[label-1]
                cv2.putText(image, f"{label_name} {score:.2f}", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (43, 204, 10), 2)

        # Create the ROS Image and publish it
        image_msg = Image()
        image_msg.data = image.tobytes()
        image_msg.height = image.shape[0]
        image_msg.width = image.shape[1]
        image_msg.encoding = 'rgb8'
        image_msg.step = 3 * image.shape[1]
        image_msg.header.stamp = rospy.Time.now()
    elif MODEL_TO_USE == "YOLO":

    # Perform object detection
        score_threshold = rospy.get_param('score_threshold', 0.2)
        t1 = time.time()
        results = cone_model.predict(image, device=0, conf=0.4, agnostic_nms=True, iou=IOU_THRESHOLD)
        results_coco = coco_model.predict(image, device=0, conf=score_threshold, agnostic_nms=True, iou=IOU_THRESHOLD)
        t2 = time.time()
        print(f"Time taken: {t2-t1:.4f}s")

        # Draw the bounding boxes
        # image_with_boxes = results[0].plot()
        image_with_boxes = results_coco[0].plot()

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

    if MODEL_TO_USE == "YOLO":
        weights_file = rospy.get_param('~weights_file', '../weights/cone_person.pt')
        cone_model = YOLO(weights_file)
        coco_model = YOLO('yolov8n.pt')
    elif MODEL_TO_USE == "FasterRCNN":
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        faster_model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)
        faster_model.eval()
        faster_model.to(device)

    with open('coco-labels-2014_2017.txt', 'r') as f:
        coco_labels = f.read().splitlines()

    # Create the publisher that will show image with bounding boxes
    publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)

    # Subscribe to the image topic
    rospy.Subscriber('/camera/color/image_raw', Image, callback, queue_size=1)

    rospy.spin()
