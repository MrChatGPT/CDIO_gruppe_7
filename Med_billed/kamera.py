import cv2
import os
import time

def capture_images(duration):
    # Define the name of the image file
    image_file = "captured_image.jpg"

    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    # Calculate end time
    end_time = time.time() + duration

    while time.time() < end_time:
        # Capture a single frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Check if the image already exists
        if os.path.exists(image_file):
            os.remove(image_file)  # Delete the existing image file
        
        # Save the captured frame to the specified file
        cv2.imwrite(image_file, frame)
        
        print(f"Image saved as {image_file}")
        
        # Wait for 0.5 second before capturing the next image
        time.sleep(0.5)

    # Release the camera
    cap.release()

# Run the function to capture images for a specified duration
capture_images(10)  # Capture images for 10 seconds
