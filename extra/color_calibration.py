import math
import os
import cv2
from cv2 import GaussianBlur
import numpy as np
import random
import imutils
import argparse
import json


def calibrateColors(image):
    """Function to calibrate the threshold values"""

    print("Press 'w' to save white thresholds.")
    print("Press 'r' to save red thresholds.")
    print("Press 'q' to save quit.")

    cv2.namedWindow('Threshold')

    # Create trackbars for threshold change
    cv2.createTrackbar('Lower Threshold', 'Threshold', 0, 255, lambda x: None)
    cv2.createTrackbar('Upper Threshold', 'Threshold',
                       255, 255, lambda x: None)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    while True:
        # Get current positions of the trackbars
        lower_thresh = cv2.getTrackbarPos('Lower Threshold', 'Threshold')
        upper_thresh = cv2.getTrackbarPos('Upper Threshold', 'Threshold')

        # Apply thresholding
        _, thresh = cv2.threshold(
            blurred, lower_thresh, upper_thresh, cv2.THRESH_BINARY_INV)

        # Display imagesq
        # cv2.imshow('Grayscale', gray)
        # cv2.imshow('Blurred', blurred)
        cv2.imshow('Threshold', thresh)
        cv2.imshow('Original', image)

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'):
            save_thresholds(lower_thresh, upper_thresh, 'white')
            print('Saved white thresholds in config.py.')
        elif key == ord('r'):
            save_thresholds(lower_thresh, upper_thresh, 'red')
            print('Saved red thresholds in config.py.')
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

def calibrateColors2(image):
    """Function to calibrate the HSV threshold values for detecting colors, specifically orange."""

    def nothing(x):
        pass

    cv2.namedWindow('HSV Calibration')

    # Creating trackbars for each HSV component
    cv2.createTrackbar('H Lower', 'HSV Calibration', 0, 179, nothing)
    cv2.createTrackbar('S Lower', 'HSV Calibration', 0, 255, nothing)
    cv2.createTrackbar('V Lower', 'HSV Calibration', 0, 255, nothing)
    cv2.createTrackbar('H Upper', 'HSV Calibration', 179, 179, nothing)
    cv2.createTrackbar('S Upper', 'HSV Calibration', 255, 255, nothing)
    cv2.createTrackbar('V Upper', 'HSV Calibration', 255, 255, nothing)

    # Convert image to HSV for better color segmentation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    while True:
        # Get current positions of the trackbars
        h_lower = cv2.getTrackbarPos('H Lower', 'HSV Calibration')
        s_lower = cv2.getTrackbarPos('L Lower', 'HSV Calibration')
        v_lower = cv2.getTrackbarPos('B Lower', 'HSV Calibration')
        h_upper = cv2.getTrackbarPos('H Upper', 'HSV Calibration')
        s_upper = cv2.getTrackbarPos('S Upper', 'HSV Calibration')
        v_upper = cv2.getTrackbarPos('V Upper', 'HSV Calibration')

        # Create the HSV range based on trackbar positions
        lower_hsv = np.array([h_lower, s_lower, v_lower], np.uint8)
        upper_hsv = np.array([h_upper, s_upper, v_upper], np.uint8)

        # Mask the image to only include colors within the specified range
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Display the original and the result side by side
        #cv2.imshow('Original', image)
        cv2.imshow('HSV Calibration', result)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Final HSV Lower:", lower_hsv)
            print("Final HSV Upper:", upper_hsv)
            break

def calibrateColorsLAB(image):
    """Function to calibrate the LAB threshold values for detecting colors."""

    def nothing(x):
        pass

    cv2.namedWindow('LAB Calibration')

    # Creating trackbars for each LAB component
    cv2.createTrackbar('L Lower', 'LAB Calibration', 0, 255, nothing)
    cv2.createTrackbar('A Lower', 'LAB Calibration', 0, 255, nothing)
    cv2.createTrackbar('B Lower', 'LAB Calibration', 0, 255, nothing)
    cv2.createTrackbar('L Upper', 'LAB Calibration', 255, 255, nothing)
    cv2.createTrackbar('A Upper', 'LAB Calibration', 255, 255, nothing)
    cv2.createTrackbar('B Upper', 'LAB Calibration', 255, 255, nothing)

    # Convert image to LAB for better color segmentation
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    while True:
        # Get current positions of the trackbars
        l_lower = cv2.getTrackbarPos('L Lower', 'LAB Calibration')
        a_lower = cv2.getTrackbarPos('A Lower', 'LAB Calibration')
        b_lower = cv2.getTrackbarPos('B Lower', 'LAB Calibration')
        l_upper = cv2.getTrackbarPos('L Upper', 'LAB Calibration')
        a_upper = cv2.getTrackbarPos('A Upper', 'LAB Calibration')
        b_upper = cv2.getTrackbarPos('B Upper', 'LAB Calibration')

        # Create the LAB range based on trackbar positions
        lower_lab = np.array([l_lower, a_lower, b_lower], np.uint8)
        upper_lab = np.array([l_upper, a_upper, b_upper], np.uint8)

        # Mask the image to only include colors within the specified range
        mask = cv2.inRange(lab, lower_lab, upper_lab)
        result = cv2.bitwise_and(image, image, mask=mask)

        # Display the calibration result
        cv2.imshow('LAB Calibration', result)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Final LAB Lower:", lower_lab)
            print("Final LAB Upper:", upper_lab)
            break

    cv2.destroyAllWindows()
    return lower_lab, upper_lab

# Example usage:
# Assuming 'image' is a loaded image in BGR format
# lower_lab, upper_lab = calibrateColors2(image)

        