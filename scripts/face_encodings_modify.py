import cv2
import numpy as np
import pickle

face_encodings = None
with open("face_encodings.pkl", "rb") as f:
    face_encodings = pickle.load(f)

# Get the list of names
names = list(face_encodings.keys())

done = False
while not done:

    print("Current list of names:")
    for i, name in enumerate(names):
        print(f"{i}: {name}")
    print("Enter the index of the name you want to delete, or 'done' to finish:")
    user_input = input()

    if user_input.lower() == "done":
        done = True
    else:
        try:
            index = int(user_input)
            if 0 <= index < len(names):
                name_to_delete = names[index]
                face_encodings.pop(name_to_delete)
                print(f"Deleted {name_to_delete} from the list.")

                names = list(face_encodings.keys())
            else:
                print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number or 'done'.")

# Save the updated face encodings
with open("face_encodings.pkl", "wb") as f:
    pickle.dump(face_encodings, f)