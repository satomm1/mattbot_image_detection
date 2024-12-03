import cv2
import face_recognition
import os
import pickle

# Directory containing images
image_directory = "../faculty"

# Dictionary to hold encodings
encodings_dict = {}

# Loop through each image in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
        # Load the image
        image_path = os.path.join(image_directory, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Get the face encodings
        face_encodings = face_recognition.face_encodings(image)
        
        # Remove the file extension
        filename = os.path.splitext(filename)[0]
        
        # Split by _ to get first and last name
        name_parts = filename.split("_")
        last_name = name_parts[0]
        first_name = name_parts[1]
        if len(name_parts) > 2:
            title = name_parts[2]
        else:
            title = None

        # Combine the parts to get the full name
        if title:
            name = f"{title} {last_name}"
        else:
            name = f"{first_name}"

        # Capitalize the first letter of each word
        name = name.title()

        # Assuming each image has only one face
        if face_encodings:
            encodings_dict[name] = face_encodings[0]
        else:
            print(f"No face found in {filename}")

# Save the encodings to a file
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(encodings_dict, f)