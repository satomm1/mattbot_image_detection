import rospy
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray, DetectedObjectWithImage, DetectedObjectWithImageArray
from std_msgs.msg import UInt32

import numpy as np
import os
import cv2

from cyclonedds.domain import DomainParticipant
from cyclonedds.topic import Topic
from cyclonedds.sub import Subscriber, DataReader
from cyclonedds.pub import Publisher, DataWriter
from cyclonedds.util import duration
from cyclonedds.idl import IdlStruct
from cyclonedds.idl.types import sequence
from cyclonedds.core import Qos, Policy, Listener
from cyclonedds.builtin import BuiltinDataReader, BuiltinTopicDcpsParticipant
from dataclasses import dataclass

@dataclass
class DataMessage(IdlStruct):
    message_type: str
    sending_agent: int
    timestamp: int
    data: str

class UnknownObjectSaver:
    """
    This class is responsible for saving unknown objects to a file.
    The unknown objects are received from the unknown_object_detector node.
    They are then identified via an LLM.
    Last, they are saved to a file to be used for training.
    """

    def __init__(self):

        # Get agent id num from environment
        self.agent_id = os.environ.get('ROBOT_ID')
        
        # Determine the names of the files already in the ../unknown_images directory
        self.unknown_images_dir = os.path.join(os.path.dirname(__file__), '../unknown_images')
        self.unknown_images = os.listdir(self.unknown_images_dir)

        # File names are in unknown_xxxxxx.jpg format, find the largest number
        if len(self.unknown_images) == 0:
            self.next_unknown_image_number = 0
        else:
            self.unknown_images.sort()
            self.unknown_images_numbers = [int(image.split('_')[1].split('.')[0]) for image in self.unknown_images]
            self.next_unknown_image_number = np.max(self.unknown_images_numbers) + 1

        print(f'Next unknown image number: {self.next_unknown_image_number}')

        # Create a subscriber to the unknown_objects topic
        rospy.Subscriber('/unknown_objects', DetectedObjectWithImageArray, self.unknown_object_callback)
        rospy.Subscriber('/send_unknown_images', UInt32, self.send_unknown_images_callback)

    def unknown_object_callback(self, msg):
        
        for detected_object_with_image in msg.objects:
            image = detected_object_with_image.data
            
            # Convert the image to a numpy array
            image = np.frombuffer(image, dtype=np.uint8).reshape(480, 640, 3)

            # Save the detected object to a file
            self.save_image(image)

    def send_unknown_images_callback(self, msg):

        # Get the agent_id to send the images to
        agent_id = msg.data

        # Create a DomainParticipant and Publisher
        participant = DomainParticipant()
        publisher = Publisher(self.participant)

        # Now publish to 'DataTopic' + agent_id topic
        topic = Topic(participant, 'DataTopic' + agent_id, DataMessage)

        # Create a DataWriter
        writer = DataWriter(publisher, topic)

        # Create a message to send
        message = DataMessage(message_type='unknown_image', sending_agent=self.agent_id, timestamp=rospy.Time.now().to_nsec(), data='')

        for i in range(self.next_unknown_image_number):
            # Load the image
            image = cv2.imread(os.path.join(self.unknown_images_dir, f'unknown_{i}.jpg'))

            # Convert the image to a string
            image_str = image.tostring()

            # Set the data field of the message
            message.data = image_str

            # Publish the message
            writer.write(message)

        # # Now delete all the images in the unknown_images directory
        # for i in range(self.next_unknown_image_number):
        #     os.remove(os.path.join(self.unknown_images_dir, f'unknown_{i}.jpg'))

        # # Now reset the next_unknown_image_number
        # self.next_unknown_image_number = 0

        # Now delete the DataWriter, Topic, Publisher, and DomainParticipant
        del writer
        del topic
        del publisher
        del participant

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
