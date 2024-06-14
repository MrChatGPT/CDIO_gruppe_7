


# import cv2
# import numpy as np

# def mask_black_objects(image):
#     # Convert the image to HSV color space
#     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
#     # Define the range for black color in HSV space
#     lower_black = np.array([0, 0, 0])
#     upper_black = np.array([180, 255, 50])
    
#     # Create a mask for black color
#     mask = cv2.inRange(hsv, lower_black, upper_black)
    
#     return mask

# # Example usage:
# # Load an image

# imagePath = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/extra/test/images/WIN_20240610_14_19_47_Pro.jpg"
# image = cv2.imread(imagePath)

# # Apply the function
# binary_mask = mask_black_objects(image)

# # Save or display the binary mask
# cv2.imwrite('binary_mask.png', binary_mask)
# cv2.imshow('Binary Mask', binary_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
from ultralytics import YOLO

# Load a pretrained YOLOv10n model
model = YOLO("yolov8n.pt")

# Perform object detection on an image

imagePath = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/extra/test/images/WIN_20240610_14_19_47_Pro.jpg"
results = model(imagePath)

# Display the results
results[0].show()