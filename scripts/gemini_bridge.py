import rospy
from mattbot_image_detection.msg import DetectedObjectWithImageArray
from image_detection_with_unknowns.msg import LabeledObject, LabeledObjectArray

import requests
import shutil
import cv2
import numpy as np
import json
"""
This script defines the GeminiBridge class, which acts as a bridge between ROS and a server hosting an LLM API.
The purpose of this bridge is to label unknown objects detected in images by sending the images to the server
and receiving the names of the most prominent objects in each bounding box.
Classes:
    GeminiBridge: A class that subscribes to a topic with unknown objects, sends the images to a server for labeling,
                  and publishes the labeled objects.
Functions:
    __init__(self, server='http://127.0.0.1:5000/gemini'): Initializes the GeminiBridge with the server URL and sets up
                                                          ROS publishers and subscribers.
    object_callback(self, msg): Callback function that processes the received images, sends them to the server for labeling,
                                and publishes the labeled objects.
    run(self): Starts the ROS node and keeps it running.
Usage:
    Run this script as a ROS node to start the GeminiBridge. The node will subscribe to the "/unknown_objects" topic,
    process the received images, send them to the server for labeling, and publish the labeled objects to the
    "/labeled_unknown_objects" topic.
"""

QUERY = """Provide the basic name of the most prominent object in each of the bounding boxes delineated by color. 
           Provide as consise of a name as possible. For example, "dog" instead of "black dog".
           If any of the bounding boxes don't have an object, do not include it in the response. Also include the caution
           level for either the object or what the object signals. The possible caution level is 'low', 'medium', and 'high'. 'low' means the object is not dangerous,
           'medium' means we should avoid a region near the object, and 'high' means we should avoid all areas near the object.
           is_static should be set to True if the object is static and does not move by itself. is_static should be set to False if the object 
           cannot move by itself.
           The possible colors of the bounding boxes are red, green, blue, purple, pink, orange, and yellow.
        """

GEMINI_COLORS = ['red', 'green', 'blue', 'purple', 'pink', 'orange', 'yellow']
COLOR_CODES = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 0, 128), (255, 192, 203), (255, 165, 0), (255, 255, 0)]

class GeminiBridge:
    
    def __init__(self, server='http://127.0.0.1:5000/gemini'):
        
        # The server to send the images to (Hosts the LLM API)
        self.server = server

        # Publisher for providing names of unknown objects
        self.labeled_pub = rospy.Publisher("/labeled_unknown_objects", LabeledObjectArray, queue_size=1)

        # Subscribe to detected unknown objects
        self.unknown_sub = rospy.Subscriber("/unknown_objects", DetectedObjectWithImageArray, self.object_callback, queue_size=1)

        self.done = False  # For testing only, only process 1 image --- remove later

        
    def object_callback(self, msg):

        # For testing only, only process 1 image --- remove later
        if self.done:
            return

        # If message is from too long ago, return
        if rospy.Time.now() - msg.header.stamp > rospy.Duration(2):
            print("Message too old")
            return
        
        # Save msg.data as an image
        np_arr = np.frombuffer(msg.data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_with_boxes = img.copy()

        num_unknowns = 0
        # Draw the bounding boxes on the image
        for i, obj in enumerate(msg.objects):

            if obj.class_name != "unknown":
                # Not an unknown object, skip
                continue

            num_unknowns += 1
            if obj.color not in GEMINI_COLORS:
                continue
            x1 = int(obj.x1)
            y1 = int(obj.y1)
            x2 = int(obj.x2)
            y2 = int(obj.y2)
            color = COLOR_CODES[GEMINI_COLORS.index(obj.color)]
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        if num_unknowns > 0:
            # Now save to the file
            cv2.imwrite("unknown_object.jpg", img_with_boxes)
            shutil.copyfile("unknown_object.jpg", f'../../../../../gemini_code/unknown_object.jpg')

            # Send the image to the LLM --- Have to send via a POST request since this version of Python doesn't 
            # support the LLM API
            data = {'query': QUERY, 'query_type': 'image', 'image_name': "unknown_object.jpg"}

            try:
                response = requests.post(self.server, json=data)
            except:
                print("Error sending image to the LLM")
                return
            response = response.json()['response']
            response = json.loads(response)

            # Store the results in a dictionary by color
            results = {}
            for obj in response:
                results[obj['bounding_box_color']] = {'object_name': obj['object_name'], 'caution_level': obj['caution_level']}
        else: 
            results = {}

        # Match the objects to the results from the LLM
        matched_object_array = LabeledObjectArray()
        matched_object_array.header.stamp = rospy.Time.now()
        matched_object_array.data = msg.data
        for obj in msg.objects:
            if obj.color in results:
                matched_object = LabeledObject()
                matched_object.class_name = results[obj.color]['object_name']
                matched_object.pose = obj.pose
                matched_object.width = obj.width
                caution_level = results[obj.color]['caution_level']
                if caution_level == 'low':
                    matched_object.caution_level = 0
                elif caution_level == 'medium':
                    matched_object.caution_level = 1
                elif caution_level == 'high':
                    matched_object.caution_level = 2
                else:
                    matched_object.caution_level = -1
                matched_object.x1 = obj.x1
                matched_object.x2 = obj.x2
                matched_object.y1 = obj.y1
                matched_object.y2 = obj.y2

                matched_object_array.objects.append(matched_object)

            elif obj.color == "none":
                matched_object = LabeledObject()
                matched_object.class_name = obj.class_name
                matched_object.pose = obj.pose
                matched_object.width = obj.width
                matched_object.caution_level = -1
                matched_object.x1 = obj.x1
                matched_object.x2 = obj.x2
                matched_object.y1 = obj.y1
                matched_object.y2 = obj.y2

                matched_object_array.objects.append(matched_object)

        # Publish the names of the new objects if any exist
        if len(matched_object_array.objects) > 0:
            self.labeled_pub.publish(matched_object_array)                    
            
        # For testing only, only process 1 image --- remove later
        self.done = True

    def run(self):
        rospy.spin()

if __name__ == "__main__":
    rospy.init_node("gemini_bridge")
    bridge = GeminiBridge()
    bridge.run()