import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import Twist
from mattbot_image_detection.msg import DetectedObject, DetectedObjectArray

import requests
import shutil

QUERY = 'Provide a list of the objects in this picture?'

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

    def object_callback(self, msg):
        if self.is_moving:
            # Only ask gemini if the object is in our path
            for obj in msg.objects:
                for pose in self.path.poses:
                    if abs(obj.pose.position.x - pose.pose.position.x) < 0.5 and abs(obj.pose.position.y - pose.pose.position.y) < 0.5:
                        # Ask gemini to label the object
                        rospy.loginfo("Asking gemini to label object")
                        # Save as a jpg
                        image_name = 'unknown_object.jpg
                        with open(image_name, 'wb') as f:
                            f.write(obj.image.data)
                        # Copy image to /gemini_code/
                        shutil.copyfile(image_name, f'../../../../../gemini_code/{image_name}')
                        data = {'query': query, 'image_name': image_name}
                        response = requests.post(url, json=data)
                        result = response.json()
                        print(result['response'])
        else:
            # Perform a check for all objects
            pass

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node("unknown_labeling")
    UnknownLabeling()
    UnknownLabeling.run()
    
