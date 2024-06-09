import cv2
import os
import time

def display_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Check if the image was loaded successfully
    if image is None:
        print("Error: Unable to load image.")
        return

    # Display image
    cv2.imshow("Image", image)

# Folder path and image name
folder_path = "/mnt/c/Users/User/OneDrive/Skrivebord/testfolder"
image_name = "photo.jpg"

# Example usage
while True:
    # Get the full path of the image
    image_path = os.path.join(folder_path, image_name)

    # Check if the image exists
    if os.path.exists(image_path):
        display_image(image_path)
    else:
        print("Error: Image not found.")

    # Wait for a key press (1 ms)
    key = cv2.waitKey(1)

    # If 'q' is pressed, break the loop
    if key == ord('q'):
        break

    # Wait for a while before checking for updates again
    time.sleep(1)
    
# Close OpenCV windows
cv2.destroyAllWindows()
