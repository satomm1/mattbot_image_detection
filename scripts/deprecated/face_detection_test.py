import face_recognition
import cv2
import time 


# # Load the known image and encode it
# known_image = face_recognition.load_image_file("obama.jpg")
# t1 = time.time()
# known_face_encoding = face_recognition.face_encodings(known_image)[0]
# t2 = time.time()
# # Load the second image to compare
# unknown_image = face_recognition.load_image_file("unknown_person.jpg")
# t3 = time.time()
# unknown_face_encodings = face_recognition.face_encodings(unknown_image)
# t4 = time.time()
# # Iterate over each face found in the unknown image
# for face_encoding in unknown_face_encodings:
#     # Check if the face matches the known face
#     results = face_recognition.compare_faces([known_face_encoding], face_encoding)
    
#     if results[0]:
#         print("Match found!")
#     else:
#         print("No match found.")


# print("Time to encode known image: ", t2 - t1)
# print("Time to encode unknown image: ", t4 - t3)


# Load image with cv2
obama = cv2.imread("obama.jpg")
small_frame = cv2.resize(obama, (0, 0), fx=0.1, fy=0.1)
face_locations = face_recognition.face_locations(small_frame)
obama_encoding = face_recognition.face_encodings(small_frame, face_locations)

t2 = time.time()
biden_image = cv2.imread("biden.jpg")
small_frame = cv2.resize(biden_image, (0, 0), fx=0.1, fy=0.1)
face_locations = face_recognition.face_locations(small_frame)
biden_encoding = face_recognition.face_encodings(small_frame, face_locations)
print(face_recognition.compare_faces([obama_encoding[0]], biden_encoding[0]))

trump_image = cv2.imread("trump.jpg")
small_frame = cv2.resize(trump_image, (0, 0), fx=0.1, fy=0.1)
face_locations = face_recognition.face_locations(small_frame)
trump_encoding = face_recognition.face_encodings(small_frame, face_locations)
print(face_recognition.compare_faces([obama_encoding[0]], trump_encoding[0]))

obama2_image = cv2.imread("obama2.jpg")
small_frame = cv2.resize(obama2_image, (0, 0), fx=0.5, fy=0.5)
face_locations = face_recognition.face_locations(small_frame)
obama2_encoding = face_recognition.face_encodings(small_frame, face_locations)
print(face_recognition.compare_faces([obama_encoding[0]], obama2_encoding[0]))
t3 = time.time()


print("Time to encode images: ", t3 - t2)


# t1 = time.time()
# known_image = face_recognition.load_image_file("obama.jpg")
# face_locations = face_recognition.face_locations(known_image, model="cnn")
# t2 = time.time()
# print("Time to find face locations: ", t2 - t1