"""
This script defines an ImageRecorder class that subscribes to a ROS image topic, detects persons in the images using 
a YOLO model, draws bounding boxes around detected persons, publishes the annotated images, and saves the images and 
bounding box coordinates to files.

Classes:
    ImageRecorder: Subscribes to a ROS image topic, detects persons using YOLO, publishes annotated images, and saves images and bounding boxes.
Functions:
    __init__(self): Initializes the ImageRecorder, sets up the YOLO model, and subscribes to the image topic.
    callback(self, data): Callback function for the image topic. Detects persons, draws bounding boxes, publishes annotated images, 
                            and saves images and bounding boxes.
    run(self): Spins the ROS node to keep it active.
Usage:
    Run the script as a ROS node to start detecting persons in images from the specified ROS image topic.
"""

import rospy
from sensor_msgs.msg import Image
import numpy as np
import time
import cv2
import os

from ultralytics import YOLO



class ImageRecorder:

    def __init__(self):
        self.old_time = time.time()

        # Get list of files in the ../images/images dir
        # files_train = os.listdir('../images/train/images')
        # files_val = os.listdir('../images/val/images')
        # files_test = os.listdir('../images/test/images')
        files = os.listdir('../images/images')

        # Get the number of files in the directory
        # self.file_count = len(files_train) + len(files_val) + len(files_test)
        self.file_count = len(files)
        print(f'Number of files: {self.file_count}')

        weights_file = rospy.get_param('~weights_file', '../weights/yolov8n.pt')
        self.model = YOLO(weights_file)
    
        # Subscribe to the image topic
        rospy.Subscriber('/camera/color/image_raw', Image, self.callback, queue_size=1)

        self.publisher = rospy.Publisher('/camera/color/image_with_boxes', Image, queue_size=1)

    def callback(self, data):
        # Get the image from the message
        image = np.frombuffer(data.data, dtype=np.uint8).reshape(data.height, data.width, -1)

        new_time = time.time()
        if new_time - self.old_time > 1:
            self.old_time = new_time

            results = self.model.predict(image, device=0, conf=0.6, agnostic_nms=True)
            
            image_width = image.shape[1]
            image_height = image.shape[0]

            person_detected = False
            bb = list()
            for ii in range(len(results[0].boxes.cls)):
            # for detected_class in results[0].boxes.cls:
                if (results[0].names[int(results[0].boxes.cls[ii])] == 'person'):
                    person_detected = True
                    
                    # Get bounding box
                    box = results[0].boxes.xyxy[ii]
                    x = (box[0].item() + box[2].item()) / 2 / image_width
                    y = (box[1].item() + box[3].item()) / 2 / image_height
                    w = (box[2].item() - box[0].item()) / image_width
                    h = (box[3].item() - box[1].item()) / image_height

                    bb.append([1, x, y, w, h])

            if person_detected:

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

                self.publisher.publish(image_msg)

                # # Random number between 0 and 1
                # rand = np.random.rand()
                # if rand < 0.7:
                #     split = 'train'
                # elif rand < 0.85:
                #     split = 'val'
                # else:
                #     split = 'test'

                # Save image to file
                # Switch channels of image
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # cv2.imwrite('../images/' + split + '/images/person' + str(self.file_count) + '.jpg', image)
                cv2.imwrite('../images/images/person' + str(self.file_count) + '.jpg', image)

                # Write bounding box to file
                # with open('../images/' + split + '/labels/person' + str(self.file_count) + '.txt', 'w') as f:
                with open('../images/labels/person' + str(self.file_count) + '.txt', 'w') as f:
                    for b in bb:
                        f.write(f'{b[0]} {b[1]} {b[2]} {b[3]} {b[4]}\n')

                self.file_count += 1
                print(f'Number of images: {self.file_count}')

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    
    # Initialize the ROS node
    rospy.init_node('image_detection', anonymous=True)

    recorder = ImageRecorder()
    recorder.run()
    

    