import rospy

import cv2
import numpy as np
import pickle
import face_recognition
import requests
import os

class FaceDetection:

    def __init__(self, cam_index=0, url='http://127.0.0.1:5000/gemini'):

        # Load the known face encodings
        self.face_encoding_file = rospy.get_param("~face_encoding_file", "face_encodings.pkl")
        self.face_encodings = self.load_face_encodings()

        self.save_new_face_dir = rospy.get_param("~save_new_face_dir", "./")

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
        num_unknown = 0
        last_unknown_time = rospy.get_time()
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
                    matches = face_recognition.compare_faces(list(self.face_encodings.values()), face_encoding, tolerance=0.52)

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

                        num_unknown = 0
                    else:
                        print("Unknown person")
                        num_unknown += 1

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

                if num_unknown > 3 and rospy.get_time() - last_unknown_time > 10:
                    # Send a message to the server
                    data = {'query': 'I don\'t recognize you, please press the \'Take Picture\' button!', 'query_type': 'face_detection'}
                    try:
                        response = requests.post(self.url, json=data)
                    except requests.exceptions.RequestException as e:
                        pass
                    
                    num_unknown = 0

                    last_unknown_time = rospy.get_time()

            # Determine if need to take an image
            data = {'query': 'need_picture', 'query_type': 'need_picture'}
            try:
                response = requests.post(self.url, json=data)
                response = response.json()
                if response['response'] == 'yes':
                    # We need to take a picture
                    print("Preparing to capture image")
                    
                    rospy.sleep(6) # Wait for 5 seconds to prepare

                    # Capture the image
                    ret, frame = self.cap.read()
                    if ret:

                        print("Captured image")
                    else:
                        print("Error: Could not capture image.")

                    # Now we need to get corresponding name
                    have_name = False
                    while not have_name:
                        # Ask for name
                        data = {'query': 'need_name', 'query_type': 'need_name'}
                        try:
                            response = requests.post(self.url, json=data)
                            response = response.json()
                            if response['response'] != 'no' and response['response'] != 'canceled':
                                name = response['response']
                                have_name = True
                                print(f"Received name: {name}")

                                # Save the image
                                save_dir = os.path.join(self.save_new_face_dir, f"{name}.jpg")
                                cv2.imwrite(save_dir, frame)

                                image = face_recognition.load_image_file(save_dir)
        
                                # Get the face encodings
                                face_encodings = face_recognition.face_encodings(image)

                                name = name.title()

                                # Assuming each image has only one face
                                if face_encodings:
                                    self.face_encodings[name] = face_encodings[0]
                                    print(f"Saved encoding for {name}")
                                else:
                                    print(f"No face found in {save_dir}")

                            elif response['response'] == 'canceled':
                                print("Image capture canceled")
                                have_name = True
                            else:
                                rospy.sleep(1)
                        except requests.exceptions.RequestException as e:
                            pass

                    
            except requests.exceptions.RequestException as e:
                pass

                    
            rate.sleep()

        self.shutdown()

    def shutdown(self):
        # Release the camera
        self.cap.release()
        cv2.destroyAllWindows()

        # Save the updated face encodings
        with open(self.face_encoding_file, "wb") as f:
            pickle.dump(self.face_encodings, f)

if __name__ == "__main__":
    rospy.init_node("face_detection")
    face_detection = FaceDetection()
    face_detection.run()