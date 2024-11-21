import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import Twist
from mattbot_image_detection.msg import DetectedObjectWithImage, DetectedObjectWithImageArray
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray

import requests
import shutil
import cv2
import numpy as np
import json

QUERY = """Provide the basic name of the most prominent object in each of the bounding boxes delineated by color. 
           Provide as consise of a name as possible. For example, "dog" instead of "black dog".
           If any of the bounding boxes appears to having nothing of note, do not include it in the response.
           The possible colors are red, green, blue, purple, pink, orange, and yellow.
        """

class UnknownLabeling:
    
    def __init__(self, url='http://127.0.0.1:5000/gemini'):

        self.url = url

        self.map_msg = rospy.wait_for_message("/navigation_map", OccupancyGrid)

        # Publisher for the labeled unknown objects
        self.labeled_pub = rospy.Publisher("/labeled_unknown_objects", DetectedObjectArray, queue_size=1)

        # Subscribe to path
        self.path = None
        self.path_sub = rospy.Subscriber("/cmd_smoothed_path", Path, self.path_callback)

        # Subscribe to the commanded velocity
        self.is_moving = False
        self.just_stopped = True
        self.cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

        # Subscribe to detected objects
        self.unknown_sub = rospy.Subscriber("/unknown_objects", DetectedObjectWithImageArray, self.object_callback, queue_size=1)

    def cmd_vel_callback(self, msg):
        if np.abs(msg.linear.x) > 0.01 or np.abs(msg.angular.z) > 0.01:
            self.is_moving = True
        else:
            if self.is_moving:
                self.just_stopped = True
            self.is_moving = False

    def path_callback(self, msg):
        self.path = msg

    # TODO Keep track of labeled objects and only ask for the ones that are not labeled

    
    def object_callback(self, msg):

        object_array = DetectedObjectArray()
        if self.is_moving and self.path is not None:
            # Only ask gemini if the object is in our path
            for obj in msg.objects:
                for pose in self.path.poses:
                    if abs(obj.pose.position.x - pose.pose.position.x) < 0.5 and abs(obj.pose.position.y - pose.pose.position.y) < 0.5:
                        # Ask gemini to label the object
                        rospy.loginfo("Asking gemini to label object")
                        # Save as a jpg
                        image_name = 'unknown_object.jpg'
                        with open(image_name, 'wb') as f:
                            f.write(obj.data)
                        # Copy image to /gemini_code/
                        shutil.copyfile(image_name, f'../../../../../gemini_code/{image_name}')
                        data = {'query': QUERY, 'query_type': 'image', 'image_name': image_name}
                        response = requests.post(self.url, json=data)
                        result = response.json()
                        print(result['response'])
        elif self.just_stopped:
            self.just_stopped = False

            # Perform a check for all objects
            # Ask gemini to label the object
            rospy.loginfo("Asking gemini to label object")
            # Save as a jpg
            image_name = 'unknown_object.jpg'
            np_arr = np.frombuffer(msg.data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Now save to the file
            cv2.imwrite(image_name, img)

            # Copy image to /gemini_code/
            shutil.copyfile(image_name, f'../../../../../gemini_code/{image_name}')
            data = {'query': QUERY, 'query_type': 'image', 'image_name': image_name}
            response = requests.post(self.url, json=data)
            result = response.json()
            result = json.loads(result['response'])
            # print(result)
            # print(result['response'])

            result_dict = dict()
            for obj in result:
                result_dict[obj['bounding_box_color']] = obj['object_name']

            for obj in msg.objects:
                obj_color = obj.color

                if obj_color not in result_dict:
                    print(f"Object with color {obj_color} not found in the response")
                else:
                    print(f"{obj_color}: {result_dict[obj_color]}")
                    detected_object = DetectedObject()
                    detected_object.width = obj.width
                    detected_object.pose = obj.pose
                    detected_object.class_name = result_dict[obj_color]
                    object_array.objects.append(detected_object)
            
            if len(object_array.objects) > 0:
                self.labeled_pub.publish(object_array)


    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("unknown_labeling")
    labeler = UnknownLabeling()
    labeler.run()
    
