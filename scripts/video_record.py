import rospy
import cv2
from sensor_msgs.msg import Image
import os
import time
import cv_bridge


class VideoRecorder:
    def __init__(self):
        rospy.init_node('video_recorder', anonymous=True)
        
        # Parameters
        self.width = rospy.get_param('~width', 640)
        self.height = rospy.get_param('~height', 480)
        self.output_path = None
        self.output_dir = rospy.get_param('~output_dir', './videos')
        self.fps = rospy.get_param('~fps', 30)

        self.bridge = cv_bridge.CvBridge()
        
        self.create_video_writer()

        # Subscribe to camera topic
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)
        
        rospy.loginfo(f"Recording video")
        rospy.loginfo(f"Saving to {self.output_path}")
        
    def create_video_writer(self):
        # Create output filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # self.output_path = os.path.join(self.output_dir, f"video_{timestamp}.mp4")
        self.output_path = f"videos/video_{timestamp}.mp4"
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, (self.width, self.height))
            
    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # Flip image upside down
            cv_image = cv2.flip(cv_image, 0)
            
            # Resize if necessary
            if cv_image.shape[1] != self.width or cv_image.shape[0] != self.height:
                cv_image = cv2.resize(cv_image, (self.width, self.height))
            
            # Write frame to video file
            self.video_writer.write(cv_image)
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
    
    def shutdown(self):
        # Release video writer resources
        try: 
            if hasattr(self, 'video_writer') and self.video_writer is not None:
                self.video_writer.release()
                rospy.loginfo(f"Video saved to {self.output_path}")
        except Exception as e:
            self.video_writer.release()

if __name__ == '__main__':
    try:
        recorder = VideoRecorder()
        # Register shutdown hook
        rospy.on_shutdown(recorder.shutdown)
        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        pass