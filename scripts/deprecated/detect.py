import rospy
from sensor_msgs.msg import Image as RosImage

import argparse
import shutil
import time
import numpy as np
import cv2
from cv_bridge import CvBridge
from PIL import Image, ImageDraw

import tflite_runtime.interpreter as tflite
from nms import non_max_suppression_yolov8

BOX_COORD_NUM = 4

def load_labels(filename):
  with open(filename, "r") as f:
    return [line.strip() for line in f.readlines()]

def callback(data):

    start_time = time.time()

    # Convert the image to a numpy array
    # np_arr = np.fromstring(data.data, np.uint8)
    # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    cv_image = bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')
    # img = cv2.resize(cv_image, (input_width, input_height))

    cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv_image_rgb)
    img = pil_image.resize((input_width, input_height))
    
    # Preprocess the image
    # image_resized = cv2.resize(image_np, (input_width, input_height))
    input_data = np.expand_dims(img, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    # Retrieve detection results
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data).transpose()
    
    # The detect output format is [x, y, width, height, class 0 score, class 1
    # score, ..., class n score]. So, if there are 80 classes, each box will be
    # an array containing 84 values (4 coords plus the 80 class scores).
    # The pose output format is similar, but instead of the class scores it has
    # a single score followed by n triples of keypoint coordinates. So if a box
    # has 17 keypoints, it will have 4 coords, plus one class score, plus 51
    # (17 * 3) keypoint coordinates, for 56 values in total.

    
    boxes = []
    for i in range(max_box_count):
        raw_box = results[i]
        center_x = raw_box[0]
        center_y = raw_box[1]
        w = raw_box[2]
        h = raw_box[3]
        class_scores = raw_box[BOX_COORD_NUM:]

        indx = np.where(class_scores > args.score_threshold)[0]
        for ii in indx:
            boxes.append([center_x, center_y, w, h, class_scores[ii], ii])
        # for index, score in enumerate(class_scores):
        #     if (score > args.score_threshold):
        #         boxes.append([center_x, center_y, w, h, score, index])    


    # Clean up overlapping boxes. See
    # https://petewarden.com/2022/02/21/non-max-suppressions-how-do-they-work/
    clean_boxes = non_max_suppression_yolov8(
        boxes, class_count, keypoint_count)
    
    # Draw the boxes on the image
    img_draw = ImageDraw.Draw(img)
    
    for box in clean_boxes:
        center_x = box[0] * input_width
        center_y = box[1] * input_height
        w = box[2] * input_width
        h = box[3] * input_height
        half_w = w / 2
        half_h = h / 2
        left_x = int(center_x - half_w)
        right_x = int(center_x + half_w)
        top_y = int(center_y - half_h)
        bottom_y = int(center_y + half_h)
        score = box[4]
        class_index = box[5]
        class_label = class_labels[class_index]
        print(f"{class_label}: {score:.2f} ({center_x:.0f}, {center_y:.0f}) {w:.0f}x{h:.0f}")
        
        img_draw.rectangle(((left_x, top_y), (right_x, bottom_y)), fill=None)
        img_draw.text((left_x, top_y), f"{class_label} {score:.2f}")
    
    # Now publish the results
    # TODO publish the results
    
    cv_image = np.array(img)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    ros_image_msg = bridge.cv2_to_imgmsg(cv_image, encoding="rgb8")
    labeled_image_publisher.publish(ros_image_msg)

    stop_time = time.time()
    
    print(f"Detection took {stop_time - start_time:.5f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect objects in a video stream.')
    parser.add_argument('--model', default="yolov8s_int8.tflite", help='Path to the TFLite model.')
    parser.add_argument('--topic', default="/camera/color/image_raw", help='Path to the labels file.')
    parser.add_argument('--score_threshold', type=float, default=0.5, help='Detection threshold.')
    parser.add_argument('--labels', default="yolov8_labels.txt", help='Path to the labels file.')
    args = parser.parse_args()

    # Load the TFLite model and allocate tensors.
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    class_labels = load_labels(args.labels)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # check the type of the input tensor
    floating_model = input_details[0]["dtype"] == np.float32

    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]

    max_box_count = output_details[0]['shape'][2]

    class_count = output_details[0]['shape'][1]
    keypoint_count = 0

    # Initialize ros node, publisher, subscriber
    rospy.init_node('detect')

    bridge = CvBridge()

    # TODO: Add publisher for detected objects and their locations
    # self.label_publisher = rospy.Publisher('labels', String, queue_size=10)
    labeled_image_publisher = rospy.Publisher('labeled_image', RosImage, queue_size=10)

    
    image_subscriber = rospy.Subscriber(args.topic, RosImage, callback, queue_size=1)
    rospy.spin()