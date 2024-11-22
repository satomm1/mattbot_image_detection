import rospy

import cv2
import numpy as np
import pickle
import face_recognition

class FaceDetection:

    def __init__(self, cam_index=0):
        self.face_encodings = self.load_face_encodings()

        # Open the camera
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set the width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set the height

        if not self.cap.isOpened():
            print("Error: Could not open the camera.")
            return

    def load_face_encodings(self):
        with open("face_encodings.pkl", "rb") as f:
            return pickle.load(f)

    def run(self):
        rate = rospy.Rate(2)
        print("Running face detection")
        while not rospy.is_shutdown():
            
            ret, frame = self.cap.read()
            if ret:
                # Resize the frame
                # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # small_frame = frame

                # Save image for debugging
                cv2.imwrite("captured_image.jpg", frame)

                # Convert the image from BGR color to RGB color
                # rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces in the current frame of video
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                for face_encoding in face_encodings:
                    # Compare the face encodings with the known encodings
                    matches = face_recognition.compare_faces(list(self.face_encodings.values()), face_encoding)

                    # If a match is found
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(self.face_encodings.keys())[first_match_index]
                        print(f"Found {name}")
                    else:
                        print("Unknown person")

            rate.sleep()

        self.shutdown()

    def shutdown(self):
        # Release the camera
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    rospy.init_node("face_detection")
    face_detection = FaceDetection()
    face_detection.run()