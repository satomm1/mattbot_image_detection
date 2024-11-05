import torch
import torch.nn as nn
import torchvision
from torchvision.ops import nms
from torchvision.models.detection import FasterRCNN
# from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.transforms import functional as F
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import time

object_threshold = 0.5
other_threshold = 0.75

# Define the custom labels:
labels_names = ['background', 'person', 'cone', 'other']

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device {device}")

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None)

# Modify output layers
num_classes = 4  # 3 classes (person, cone, other) + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor.cls_score = nn.Linear(in_features=in_features, out_features=num_classes, bias=True)
model.roi_heads.box_predictor.bbox_pred = nn.Linear(in_features=in_features, out_features=num_classes*4, bias=True)

# Load custom weights
model.load_state_dict(torch.load('../weights/openset_mobilenet_small_not_frozen.pth'))

# Set to evaluation mode and send to device
model.eval()
model.to(device)

# Initialize the ROS node
rospy.init_node('object_detection_node')

# Create a publisher to publish the detected objects
pub = rospy.Publisher('/detected_objects', Image, queue_size=1)
bridge = CvBridge()

# Define the callback function for the image topic
def image_callback(msg):

    t1 = time.time()
    # Convert the ROS Image message to OpenCV format
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

    # Rotate image 180 degrees
    cv_image = cv2.rotate(cv_image, cv2.ROTATE_180)

    # Convert the image to a PyTorch tensor
    image_tensor = F.to_tensor(cv_image)

    # Add a batch dimension to the tensor
    image_tensor = image_tensor.unsqueeze(0)

    # Run the image through the model
    
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)

    # Draw bounding boxes on the image
    cv_image = cv_image.copy()

    # print(predictions[0])
    boxes = predictions[0]['boxes']
    scores = predictions[0]['scores']
    labels = predictions[0]['labels']

    # Apply NMS
    iou_threshold = 0.4  # Set the IoU threshold for NMS
    keep = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep].cpu().numpy()
    labels = labels[keep].cpu().numpy()
    scores = scores[keep].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if label != 3 and score > object_threshold:
            box = [int(i) for i in box]
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_name = labels_names[label]
            cv2.putText(cv_image, label_name + " " + str(round(score,2)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if label == 3 and score > other_threshold:
            box = [int(i) for i in box]
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            label_name = labels_names[label]
            cv2.putText(cv_image, label_name + " " + str(round(score,2)), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    # Convert the image back to a ROS Image message
    annotated_image_msg = bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')

    # Publish the annotated image
    pub.publish(annotated_image_msg)

    t2 = time.time()
    print(f"Time taken: {t2-t1:.4f}s")

    # Get time of original image message:
    time_original = msg.header.stamp
    # Compare to current time
    time_now = rospy.Time.now()
    time_diff = time_now - time_original
    # Convert to seconds
    time_diff = time_diff.to_sec()
    print(f"Time difference: {time_diff:.4f}s")

# Subscribe to the image topic
rospy.Subscriber('/camera/color/image_raw', Image, image_callback, queue_size=1)

# Spin the ROS node
rospy.spin()