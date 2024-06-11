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
        print("Vi starter kamera")
        self.cap = cv2.VideoCapture(0)  # 0 is the default camera

        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._run_video)
        self.thread.start()

    def _run_video(self):
        #print("Video started. Press 'q' to stop.")
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
            return frame

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

        return frame

    def release_camera(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        print("Camera released.")


