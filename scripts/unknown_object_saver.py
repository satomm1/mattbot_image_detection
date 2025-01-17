import rospy
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray

import numpy as np
import os
import cv2

class UnknownObjectSaver:
    """
    This class is responsible for saving unknown objects to a file.
    The unknown objects are received from the unknown_object_detector node.
    They are then identified via an LLM.
    Last, they are saved to a file to be used for training.
    """

    def __init__(self):
        
        # Determine the names of the files already in the ../unknown_images directory
        self.unknown_images_dir = os.path.join(os.path.dirname(__file__), '../unknown_images')
        self.unknown_images = os.listdir(self.unknown_images_dir)

        # File names are in unknown_xxxxxx.jpg format, find the largest number
        self.unknown_images.sort()
        self.unknown_images_numbers = [int(image.split('_')[1].split('.')[0]) for image in self.unknown_images]
        self.next_unknown_image_number = max(self.unknown_images_numbers) + 1


        # Create a subscriber to the unknown_objects topic
        rospy.Subscriber('/unknown_objects', DetectedObjectWithImageArray, self.unknown_object_callback)

    def unknown_object_callback(self, msg):
        
        for detected_object_with_image in msg.objects:
            image = detected_object_with_image.data
            
            # Convert the image to a numpy array
            image = np.frombuffer(image, dtype=np.uint8).reshape(480, 640, 3)


            # Save the detected object to a file
            self.save_unknown_object(detected_object, detected_object_image)

    def save_image(self, image, filename=None):
        if filename is None:
            filename = f'unknown_{self.next_unknown_image_number}.jpg'
            self.next_unknown_image_number += 1

        cv2.imwrite(os.path.join(self.unknown_images_dir, filename), image)

    
    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("unknown_object_saver")

    unknown_object_saver = UnknownObjectSaver()
    unknown_object_saver.run()
