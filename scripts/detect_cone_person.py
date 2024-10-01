import rospy
from sensor_msgs.msg import Image

from ultralytics import YOLO
import numpy as np

def callback(data):
    # Get the image from the message
    image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

    # Perform object detection
    results = model.predict(image, device=0, conf=0.4, agnostic_nms=True)

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


if __name__ == '__main__':
    
    # Initialize the ROS node
    rospy.init_node('image_detection', anonymous=True)

    weights_file = rospy.get_param('~weights_file', '../weights/cone_person.pt')
    model = YOLO(weights_file)

    # Create the publisher that will show image with bounding boxes
    publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)

    # Subscribe to the image topic
    rospy.Subscriber('/camera/color/image_raw', Image, callback, queue_size=1)

    rospy.spin()
