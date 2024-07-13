import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.transforms import functional as F
import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device {device}")

# Load the pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
model.to(device)

# Initialize the ROS node
rospy.init_node('object_detection_node')

# Create a publisher to publish the detected objects
pub = rospy.Publisher('/detected_objects', Image, queue_size=10)
bridge = CvBridge()

# Define the callback function for the image topic
def image_callback(msg):
    # Convert the ROS Image message to OpenCV format
    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

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

    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        if score > 0.5:
            box = [int(i) for i in box]
            cv2.rectangle(cv_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(cv_image, f'{label.item()}', (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)            

    # Convert the image back to a ROS Image message
    annotated_image_msg = bridge.cv2_to_imgmsg(cv_image, encoding='rgb8')

    # Publish the annotated image
    pub.publish(annotated_image_msg)

# Subscribe to the image topic
rospy.Subscriber('/camera/color/image_raw', Image, image_callback, queue_size=1)

# Spin the ROS node
rospy.spin()