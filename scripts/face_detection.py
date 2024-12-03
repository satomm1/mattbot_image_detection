import rospy

import cv2
import numpy as np
import pickle
import face_recognition
import requests

class FaceDetection:

    def __init__(self, cam_index=0, url='http://127.0.0.1:5000/gemini'):

        # Load the known face encodings
        self.face_encoding_file = rospy.get_param("~face_encoding_file", "face_encodings.pkl")
        self.face_encodings = self.load_face_encodings()

        # URL to send the query to
        self.url = url

        # Get obama file location
        self.obama_file = rospy.get_param("~obama_file", "obama.jpg")

        # Open the camera
        self.cap = cv2.VideoCapture(cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set the width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set the height

        if not self.cap.isOpened():
            print("Error: Could not open the camera.")
            return

    def load_face_encodings(self):
        with open(self.face_encoding_file, "rb") as f:
            return pickle.load(f)

    def run(self):
        rate = rospy.Rate(2)

        # Load an image and encode it since first encoding takes a long time
        obama = cv2.imread(self.obama_file)
        small_frame = cv2.resize(obama, (0, 0), fx=0.1, fy=0.1)
        face_locations = face_recognition.face_locations(small_frame)
        obama_encoding = face_recognition.face_encodings(small_frame, face_locations)


        print("Running face detection")

        detect_counts = {}
        while not rospy.is_shutdown():
            
            ret, frame = self.cap.read()
            if ret:
                # Resize the frame
                # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # small_frame = frame

                # Save image for debugging
                # cv2.imwrite("captured_image.jpg", frame)

                # Convert the image from BGR color to RGB color
                # rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces in the current frame of video
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                names_found = []
                for face_encoding in face_encodings:
                    # Compare the face encodings with the known encodings
                    matches = face_recognition.compare_faces(list(self.face_encodings.values()), face_encoding)

                    # If a match is found
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(self.face_encodings.keys())[first_match_index]
                        print(f"Found {name}")
                        names_found.append(name)

                        if name in detect_counts:
                            detect_counts[name] += 1
                        elif name not in detect_counts:
                            detect_counts[name] = 1
                    else:
                        print("Unknown person")

                name_three_times = []
                for name in list(detect_counts.keys()):
                    
                    if name not in names_found:
                        detect_counts[name] = 0
                    elif detect_counts[name] == 3:
                        name_three_times.append(name)

                if name_three_times:
                    if len(name_three_times) == 1:
                        query = f"Hello, {name_three_times[0]}."
                    elif len(name_three_times) == 2:
                        query = f"Hello, {name_three_times[0]} and {name_three_times[1]}."
                    else:
                        query = f"Hello, {', '.join(name_three_times[:-1])}, and {name_three_times[-1]}."
                    
                    print(query)

                    data = {'query': query, 'query_type': 'face_detection'}
                    try:
                        response = requests.post(self.url, json=data)
                    except requests.exceptions.RequestException as e:
                        pass
                    
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