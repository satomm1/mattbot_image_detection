import cv2

def capture_image():
    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    # Capture a single frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        return

    # Save the captured image
    cv2.imwrite("captured_image.jpg", frame)
    print("Image captured and saved as 'captured_image.jpg'")

    # Release the camera
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()