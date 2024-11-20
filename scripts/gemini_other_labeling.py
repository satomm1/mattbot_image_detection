import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import Twist
from mattbot_image_detection.msg import DetectedObjectWithImage, DetectedObjectWithImageArray

import requests
import shutil
import cv2
import numpy as np

QUERY = 'Provide the basic name of the most prominent object in this image. Provide as consise of a name as possible. For example, "dog" instead of "black dog".'

class UnknownLabeling:
    
    def __init__(self, url='http://127.0.0.1:5000/gemini'):

        self.url = url

        self.map_msg = rospy.wait_for_message("/navigation_map", OccupancyGrid)

        # Subscribe to path
        self.path = None
        self.path_sub = rospy.Subscriber("/cmd_smoothed_path", Path, self.path_callback)

        # Subscribe to the commanded velocity
        self.is_moving = False
        self.cmd_vel_sub = rospy.Subscriber("/cmd_vel", Twist, self.cmd_vel_callback)

        # Subscribe to detected objects
        self.unknown_sub = rospy.Subscriber("/unknown_objects", DetectedObjectWithImageArray, self.object_callback)

    def cmd_vel_callback(self, msg):
        if msg.linear.x > 0.01 or msg.angular.z > 0.01:
            self.is_moving = True
        else:
            self.is_moving = False

    def path_callback(self, msg):
        self.path = msg

    # TODO Keep track of labeled objects and only ask for the ones that are not labeled
    def object_callback(self, msg):
        if self.is_moving:
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
        else:
            # Perform a check for all objects
             for obj in msg.objects:

                # Ask gemini to label the object
                rospy.loginfo("Asking gemini to label object")
                # Save as a jpg
                image_name = 'unknown_object.jpg'
                np_arr = np.frombuffer(obj.data, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                # Now save to the file
                cv2.imwrite(image_name, img)

                # Copy image to /gemini_code/
                shutil.copyfile(image_name, f'../../../../../gemini_code/{image_name}')
                data = {'query': QUERY, 'query_type': 'image', 'image_name': image_name}
                response = requests.post(self.url, json=data)
                result = response.json()
                print(result['response'])

                # Shutdown the node
                rospy.signal_shutdown("All objects labeled")

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("unknown_labeling")
    labeler = UnknownLabeling()
    labeler.run()
    
