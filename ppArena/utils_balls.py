from calendar import c
import math
import os
from re import I
from secrets import randbelow
from tabnanny import verbose
from turtle import up
import cv2
from cv2 import GaussianBlur
import numpy as np
# from config import *
import random
import imutils
from imutils import paths
import argparse
from skimage import exposure
import json
# from utils import *
#from arena import perspectiveTransDyn




def circle_detection(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply a blur
    """ 
    *Parameter one, gray, is the source image converted to grayscale. 
    Making the algorithms that operate on the image computationally less intensive.
    **Parameter two and three is the kernel size. Which determines the width and height of the Gaussian filter. 
    A kernel size of (9, 9) means that the filter window is 9 pixels by 9 pixels. 
    The larger the kernel, the stronger the blur effect.
    
    ***The standard deviation in the X and Y directions; when set to 0, it is calculated from the kernel size. 
    A higher value for standard deviation means more blur.
    """
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)  
    # cv2.imshow('Detected Balls', gray_blurred)
    
    # Perform Hough Circle Transform (Detect circles)
    """
    *cv2.HOUGH_GRADIENT, this method is the only one available for circle detection in OpenCV and uses the gradient information of the image.
    
    **dp=1 means the accumulator has the same resolution as the input image. 
    If dp is greater, the accumulator resolution is reduced, and vice versa.
   
    ***minDist=40: The minimum distance between the centers of detected circles.
    
    ****param1=50: The higher threshold of the two passed to the Canny edge detector (the lower one is half of this). It's used in the edge detection stage.
    *****param2=30: The accumulator threshold for the circle centers at the detection stage. 
    The smaller it is, the more false circles may be detected. Circles with an accumulator value above this threshold are returned.
    ******minRadius= and maxRadius=: The minimum and maximum radius of the circles to be detected.
    """

    #mindist=18
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=18, 
                               param1=50, param2=24, minRadius=12, maxRadius=17) #minRadius=5, maxRadius=20 , param2= 28 ORIG
  ##
     # List to store circle data
    stored_circles = []
     # Filter out the circles that correspond to the ping pong balls
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]  # x, y center and radius of circle
            
            # Draw the outer circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 1) #2
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 0), 2)
            
            # Put text 'Ball' near the detected ball
            # cv2.putText(image, 'Ball', (x - r, y - r),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # print(f"'center': {x, y}, 'radius': {r}")
            # Store the circles data
            stored_circles.append({'center': (x, y), 'radius': r, 'label': 'Ball'})
            save_balls(stored_circles)
    # with open('stored_circles.json', 'w') as file:
    #     json.dump(stored_circles, file, indent=4)
    # Display the result
    # cv2.imshow('Detected Balls', image)
    return image, stored_circles
    # return image
  ##
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    ################################

    # # Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)
    # # Blur using 3 * 3 kernel
    # gray_blurred = cv2.blur(gray, (3, 3))
    # cv2.imshow('gray blurred', gray_blurred)
    # # Apply Hough transform on the blurred image
    # detected_circles = cv2.HoughCircles(gray_blurred,
    #                                     cv2.HOUGH_GRADIENT, 1, 20,
    #                                     param1=50, param2=30,
    #                                     minRadius=1, maxRadius=40)
    # # Draw circles that are detected
    # if detected_circles is not None:
    #     # Convert the circle parameters a, b and r to integers
    #     detected_circles = np.uint16(np.around(detected_circles))
    #     for pt in detected_circles[0, :]:
    #         a, b, r = pt[0], pt[1], pt[2]
    #         # Draw the circumference of the circle
    #         cv2.circle(image, (a, b), r, (0, 255, 0), 2)
    #         # Draw a small circle (of radius 1) to show the center
    #         cv2.circle(image, (a, b), 1, (0, 0, 255), 3)
    #         cv2.imshow("Detected Circle", image)
    #         cv2.waitKey(0)
    # # Close the window when done
    # cv2.destroyAllWindows()


def save_balls(circles, filename="balls.json"):
    balls = [(int(circle['center'][0]), int(circle['center'][1])) for circle in circles]
    with open(filename, 'w') as file:
        json.dump(balls, file, indent=4)


def saveOrange_balls(balls, filename="orangeballs.json"):
    # balls = [(int(circle['center'][0]), int(circle['center'][1])) for circle in circles]
    with open(filename, 'w') as file:
        json.dump(balls, file, indent=4)

def saveWhite_balls(balls, filename="whiteballs.json"):
    # balls = [(int(circle['center'][0]), int(circle['center'][1])) for circle in circles]
    with open(filename, 'w') as file:
        json.dump(balls, file, indent=4)
        
def load_balls(filename="balls.json"):
    with open(filename, 'r') as file:
        balls = json.load(file)
    # Convert the list of lists back to a list of tuples
    return [tuple(center) for center in balls]

def print_balls(filename="balls.json"):
    balls = load_balls(filename)
    for ball in balls:
        print(ball)



# Function to check if a point is within any detected orange region
def check_point_in_orange_region(contours):
    # print_balls("balls.json")
    
    #To store balls in separate arrays
    white_balls = []
    orange_balls = []

    # Check each ball coordinate
    balls = load_balls("balls.json")
    for px, py in balls:
        point_in_orange_region = False
        for contour in contours:
            # Check if the point (px, py) is inside this contour
            dist = cv2.pointPolygonTest(contour, (px, py), False)
            if dist >= 0:
                # print(f"dist in pointPolygonTest is: {dist}.\n The point is ({px}, {py}). IN IF")
                point_in_orange_region = True
                break  # Exit the loop if the point is found in any contour
        # print(f"dist in pointPolygonTest is: {dist}.\n The point is ({px}, {py}). NOT IN IF")
        # print(f"dist {dist}")
        if point_in_orange_region:
            print(f"The point ({px}, {py}) is within an orange region.")
            orange_balls.append((px, py))
        else:
            print(f"The point ({px}, {py}) is not within any orange region.")
            white_balls.append((px, py))

    saveOrange_balls(orange_balls)
    saveWhite_balls(white_balls)

