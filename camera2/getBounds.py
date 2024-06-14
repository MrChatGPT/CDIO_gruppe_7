import cv2
import numpy as np

def nothing(x):
    pass

def mask_dark_matte_gray_with_contours(image):
    # Create a window
    cv2.namedWindow('Mask Adjustments')

    # Create trackbars for adjusting HSV values
    cv2.createTrackbar('H Lower', 'Mask Adjustments', 0, 180, nothing)
    cv2.createTrackbar('S Lower', 'Mask Adjustments', 0, 255, nothing)
    cv2.createTrackbar('V Lower', 'Mask Adjustments', 0, 255, nothing)
    cv2.createTrackbar('H Upper', 'Mask Adjustments', 180, 180, nothing)
    cv2.createTrackbar('S Upper', 'Mask Adjustments', 50, 255, nothing)
    cv2.createTrackbar('V Upper', 'Mask Adjustments', 100, 255, nothing)

    while True:
        # Get the current positions of the trackbars
        h_lower = cv2.getTrackbarPos('H Lower', 'Mask Adjustments')
        s_lower = cv2.getTrackbarPos('S Lower', 'Mask Adjustments')
        v_lower = cv2.getTrackbarPos('V Lower', 'Mask Adjustments')
        h_upper = cv2.getTrackbarPos('H Upper', 'Mask Adjustments')
        s_upper = cv2.getTrackbarPos('S Upper', 'Mask Adjustments')
        v_upper = cv2.getTrackbarPos('V Upper', 'Mask Adjustments')

        # Define the range for dark matte gray in HSV space based on the trackbar positions
        lower_gray = np.array([h_lower, s_lower, v_lower])
        upper_gray = np.array([h_upper, s_upper, v_upper])

        # Convert the image to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create a mask for dark matte gray color
        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # Find contours
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw the contours on the original image
        contour_image = image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        # Display the original image, the binary mask, and the contour image
        cv2.imshow('Original Image', image)
        cv2.imshow('Binary Mask', mask)
        cv2.imshow('Contours', contour_image)

        # Break the loop when 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Destroy all windows
    cv2.destroyAllWindows()

# Example usage:
# Load an image
imagePath = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/camera2/testImg1.jpg"

image = cv2.imread(imagePath)

# Apply the function
mask_dark_matte_gray_with_contours(image)
