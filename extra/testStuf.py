import cv2
import numpy as np

# Create a blank image
image = np.zeros((400, 400, 3), dtype=np.uint8)

# Create a contour (example contour for demonstration)
contour = np.array([
    [50, 50],
    [200, 50],
    [50, 200],
    [200, 200],
    [125, 125]
], dtype=np.int32)

# Ensure the contour is in the shape (n, 1, 2) as expected by OpenCV functions
contour = contour.reshape((-1, 1, 2))

# Fit an ellipse to the contour
if len(contour) >= 5:  # fitEllipse requires at least 5 points
    ellipse = cv2.fitEllipse(contour)
    # Draw the ellipse
    cv2.ellipse(image, ellipse, (0, 255, 0), 2)  # Drawing with a green color

# Display the result
cv2.imshow("Fitted Ellipse", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
