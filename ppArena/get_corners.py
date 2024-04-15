import cv2
import numpy as np

# Initialize a list to store the coordinates of the corners
clicked_points = []

def draw_circle(event, x, y, flags, param):
    global clicked_points, img
    if event == cv2.EVENT_LBUTTONDOWN:
        # Save the coordinates of the click event
        clicked_points.append((x, y))
        # Draw a small circle at the point of click
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)
        # Show the image with the circle drawn
        cv2.imshow('Image', img)

# Read in your image
img = cv2.imread('test/images/WIN_20240410_10_31_07_Pro.jpg')

# Set up the window and bind the function to window mouse events
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_circle)

while True:
    # Display the image
    cv2.imshow('Image', img)
    key = cv2.waitKey(1) & 0xFF
    # Break the loop when 'c' is pressed
    if key == ord('c'):
        break

# Check if we have exactly four points
if len(clicked_points) == 4:
    input_corners = np.float32(clicked_points)
    print(input_corners)
cv2.destroyAllWindows()
