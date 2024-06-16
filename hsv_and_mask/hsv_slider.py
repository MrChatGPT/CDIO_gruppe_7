import cv2
import numpy as np

def nothing(x):
    pass

def create_trackbars():
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('HMin', 'Trackbars', 0, 179, nothing)
    cv2.createTrackbar('SMin', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('VMin', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('HMax', 'Trackbars', 0, 179, nothing)
    cv2.createTrackbar('SMax', 'Trackbars', 0, 255, nothing)
    cv2.createTrackbar('VMax', 'Trackbars', 0, 255, nothing)

    # Set initial positions of the trackbars to the mid-range values
    cv2.setTrackbarPos('HMax', 'Trackbars', 179)
    cv2.setTrackbarPos('SMax', 'Trackbars', 255)
    cv2.setTrackbarPos('VMax', 'Trackbars', 255)

def get_trackbar_values():
    h_min = cv2.getTrackbarPos('HMin', 'Trackbars')
    s_min = cv2.getTrackbarPos('SMin', 'Trackbars')
    v_min = cv2.getTrackbarPos('VMin', 'Trackbars')
    h_max = cv2.getTrackbarPos('HMax', 'Trackbars')
    s_max = cv2.getTrackbarPos('SMax', 'Trackbars')
    v_max = cv2.getTrackbarPos('VMax', 'Trackbars')
    return (h_min, s_min, v_min), (h_max, s_max, v_max)

def interactive_hsv_tuning(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    create_trackbars()

    while True:
        lower, upper = get_trackbar_values()
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
        result = cv2.bitwise_and(image, image, mask=mask)

        combined = np.hstack((image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result))
        cv2.imshow('Tuning', combined)

        if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
            break

    cv2.destroyAllWindows()
    return lower, upper

# Example usage
image_path = 'newest_images/im5.jpg'
lower_hsv, upper_hsv = interactive_hsv_tuning(image_path)
print(f'Lower HSV: {lower_hsv}')
print(f'Upper HSV: {upper_hsv}')
