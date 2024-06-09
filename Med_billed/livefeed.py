import cv2
import os
import time
import threading

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None

    def start_video(self):
        # Initialize the camera
        self.cap = cv2.VideoCapture(0)  # 0 is the default camera

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_video)
        self.thread.start()

    def _run_video(self):
        print("Video started. Press 'q' to stop.")
        while self.running:
            # Capture frame-by-frame
            ret, frame = self.cap.read()

            if not ret:
                print("Error: Could not read frame.")
                break

            # Display the resulting frame
            cv2.imshow('Live Feed', frame)

            # Press 'q' on the keyboard to exit the video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        # Close the window when done
        cv2.destroyAllWindows()

    def capture_image(self):
        if self.cap is None or not self.cap.isOpened():
            print("Error: Camera is not initialized.")
            return

        # Capture a single frame
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Could not read frame.")
            return

        # Define the name of the image file
        image_file = "captured_image.jpg"

        # Check if the image already exists
        if os.path.exists(image_file):
            os.remove(image_file)  # Delete the existing image file

        # Save the captured frame to the specified file
        cv2.imwrite(image_file, frame)
        print(f"Image saved as {image_file}")

    def release_camera(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        print("Camera released.")

# Create a CameraHandler instance
camera_handler = CameraHandler()

# Start video in a separate thread
camera_handler.start_video()

# Main loop to interact with the camera handler
try:
    while True:
        command = input("Enter 'capture' to take a picture or 'exit' to quit: ").strip().lower()
        if command == 'capture':
            camera_handler.capture_image()
        elif command == 'exit':
            break
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
