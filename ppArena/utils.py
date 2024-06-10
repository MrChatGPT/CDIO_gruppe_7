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

#from arena import perspectiveTransDyn






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
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    while True:
        # Get current positions of the trackbars
        h_lower = cv2.getTrackbarPos('H Lower', 'HSV Calibration')
        s_lower = cv2.getTrackbarPos('S Lower', 'HSV Calibration')
        v_lower = cv2.getTrackbarPos('V Lower', 'HSV Calibration')
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



def getImage():
    """This is just a dummy function. It will be replaced by the camera module."""
    
    # image = cv2.imread('test/images/WIN_20240403_10_40_59_Pro.jpg')
    # image = cv2.imread('test/images/WIN_20240403_10_39_46_Pro.jpg') 
    # image = cv2.imread('test/images/WIN_20240403_10_40_38_Pro.jpg') #hvid nej
    # image = cv2.imread('test/images/WIN_20240403_10_40_58_Pro.jpg') 
    # image = cv2.imread('test/images/pic50upsidedown.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_31_43_Pro.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_31_07_Pro.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_31_07_Pro.jpg') #orig pic with transfrom new
    # image = cv2.imread('test/images/pic50egghorizontal.jpg') 
    # image = cv2.imread('test/images/WIN_20240410_10_30_54_Pro.jpg') 
    image = cv2.imread('test/images/WIN_20240610_09_33_12_Pro.jpg') 
    
    return image

#THE REAL FUNCTION
# def arena_draw(image, x, y, w, h, area):
    # Start coordinate, here (x, y), represents the top left corner of rectangle 
    start_point = (x, y)
    
    # End coordinate, here (x+w, y+h), represents the bottom right corner of rectangle
    end_point = (x+w, y+h)
    
    # Green color in BGR
    color = (0, 255, 0)  # Using a standard green color; modify as needed
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.rectangle() method to draw a rectangle around the car
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    # Optionally, add text label if needed
    cv2.putText(image, 'arena', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return image


#TESTING
def goal_draw(image, x, y):
 
    
    # Coordinates for the goal rectangle
    ##LEFT GOAL
    goal_x = x+20
    goal_y = y+430
    goal_w = 22
    goal_h = 130
  
    start_point_goal = (goal_x, goal_y)
    end_point_goal = (goal_x + goal_w, goal_y + goal_h)
    color_goal = (0, 255, 0)
    thickness_goal = 2
    
    image = cv2.rectangle(image, start_point_goal, end_point_goal, color_goal, thickness_goal)
    cv2.putText(image, 'L', (goal_x, goal_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_goal, thickness_goal)



    # Coordinates for the goal rectangle
    ##RIGHT GOAL
    goal_x = x+1360
    goal_y = y+470
    goal_w = 22
    goal_h = 75
    


    start_point_goal = (goal_x, goal_y)
    end_point_goal = (goal_x + goal_w, goal_y + goal_h)
    color_goal = (0, 255, 0)
    thickness_goal = 2
    
    image = cv2.rectangle(image, start_point_goal, end_point_goal, color_goal, thickness_goal)
    cv2.putText(image, 'R', (goal_x, goal_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_goal, thickness_goal)

    # # To display the image
    # cv2.imshow('Result', image)
    
    return image





#Used for the cross (and arena), but not limited to
def square_draw(image, x, y, w, h, area):
    # # Start coordinate, here (x, y)
    # start_point = (x+60, y)
    
    # # End coordinate
    # end_point = (x+60,y+h)
    
    # # Green color in BGR
    # color = (0, 255, 0)  # Using a standard green color; modify as needed
    
    # # Line thickness of 2 px
    # thickness = 2
    
    # # Using cv2.rectangle() method to draw a rectangle around the car
    # image = cv2.line(image, start_point, end_point, color, thickness)
    
    # # Optionally, add text label if needed
    # cv2.putText(image, 'cross', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    

    # image = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2) 


    # print("before recreating the points")

     # Recreate the rectangle points from x, y, w, h
    rect_points = np.array([
        [x, y],
        [x + w, y],
        [x, y + h],
        [x + w, y + h]
    ], dtype=np.float32)

    # print(f"rect_points={rect_points}")
    # print("after recreating the points")
    # The points need to be ordered correctly for minAreaRect to work
    rect_points = cv2.convexHull(rect_points)

    # Find the minimum area rectangle
    min_area_rect = cv2.minAreaRect(rect_points)

    """ Cross
    minimum area rectangle=((1100.0, 523.5), (167.0, 168.0), 90.0)
    The first two numbers (1100.0, 523.5), shows the x and y-axis of the central point in the square.
    """
    # print(f"minimum area rectangle={min_area_rect}") 

    # Convert the rectangle to box points (four corners)
    """
    Box prints out the the coordinates respectively:
    box=[xTopleft,yTopleft],
        [xTopright, yTopright],
        [xBottomright, yBottomright],
        [xBottomleft,yBottomleft]
    """
    box = cv2.boxPoints(min_area_rect)
    # print(f"box={box}")
    box = np.int0(box)
 



    return box, min_area_rect



def line_draw(image, x, y, w, h, area):

    # Green color in BGR 
    color = (0, 255, 0) 
    
    # Line thickness of 9 px 
    thickness = 9
 


    # represents the top left corner of image 
    start_point = (x, y) 
    # represents the top right corner of image 
    end_point = (x+w, y) 
    # Draw a diagonal green line with thickness of 9 px 
    image = cv2.line(image, start_point, end_point, color, thickness) 




    # represents the top left corner of image 
    start_point = (x, y) 
    # represents the bottom left corner of image 
    end_point = (x, y+h) 
    # Draw a diagonal green line
    image = cv2.line(image, start_point, end_point, color, thickness) 



    # represents the top right corner of image 
    start_point = (x+w, y) 
    # represents the bottom right corner of image 
    end_point = (x+w, y+h) 
    # Draw a diagonal green line
    image = cv2.line(image, start_point, end_point, color, thickness) 

    # represents the bottom left corner of image 
    start_point = (x, y+h) 
    # represents the bottom right corner of image 
    end_point = (x+w, y+h) 
    # Draw a diagonal green line
    image = cv2.line(image, start_point, end_point, color, thickness) 

    return image



def car_draw(image, x, y, w, h, area):
    # Start coordinate, here (x, y), represents the top left corner of rectangle 
    start_point = (x, y)
    
    # End coordinate, here (x+w, y+h), represents the bottom right corner of rectangle
    end_point = (x+w, y+h)
    
    # Green color in BGR
    color = (0, 255, 0)  # Using a standard green color; modify as needed
    
    # Line thickness of 2 px
    thickness = 2
    
    # Using cv2.rectangle() method to draw a rectangle around the car
    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    # Optionally, add text label if needed
    cv2.putText(image, 'Car', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    
    return image


def egg_draw(image, x, y, w, h, area):
    #https://www.geeksforgeeks.org/python-opencv-cv2-ellipse-method/
    # Calculate center coordinates 
    center_coordinates = (x + w//2, y + h//2)
    
    # Define axes length
    axesLength = (w//2, h//2)
 
    
    # Ellipse parameters
    angle = 0
    startAngle = 0
    endAngle = 360
    
    # Draw the ellipse on the image
    image = cv2.ellipse(image, center_coordinates, axesLength, 
                        angle, startAngle, endAngle, (0, 255, 0), 3)
    
    # Put text 'Egg' near the detected egg
    cv2.putText(image, 'Egg', (center_coordinates[0] - 10, center_coordinates[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return image


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
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=50, param2=28, minRadius=10, maxRadius=20) #minRadius=5, maxRadius=20

  ##
     # List to store circle data
    stored_circles = []
     # Filter out the circles that correspond to the ping pong balls
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]  # x, y center and radius of circle
            
            # Draw the outer circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 0), 2)
            
            # Put text 'Ball' near the detected ball
            cv2.putText(image, 'Ball', (x - r, y - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            
            # Store the circles data
            stored_circles.append({'center': (x, y), 'radius': r, 'label': 'Ball'})

    # with open('stored_circles.json', 'w') as file:
    #     json.dump(stored_circles, file, indent=4)
    save_balls(stored_circles)
    hello = load_balls()
    # Display the result
    # cv2.imshow('Detected Balls', image)
    return image, stored_circles


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


#TRUE
def detect_ball_colors(image):
    #https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/
    #https://colorpicker.me/#ffffff
    # https://colorizer.org/
  # Capturing video through webcam 
 #  webcam = cv2.VideoCapture(0) 
  
 
    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    
    # Set range for red color and  
    # define mask 
    red_lower = np.array([0, 113, 180], np.uint8) #HSV   0, 113, 180 # 6, 128, 244
    red_upper = np.array([9, 255, 255], np.uint8) #HSV  9, 255, 255 # 10, 163, 255
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 


    orange_lower = np.array([11, 121, 215], np.uint8) #HSV
    orange_upper = np.array([65, 211, 255], np.uint8) #HSV
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 


    """
    Attempt on the picture ending with 59, with both higher and lower values at the same time
    BEST SO FAR
    """
    #ORIGINAL
    white_lower = np.array([ 0, 0, 209], np.uint8) #HSV 6, 0, 191
    white_upper = np.array([100, 75, 255], np.uint8) #HSV 179, 42, 255
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 

    

    """
    Attempt on the picture ending with 46
    """
    blue_lower = np.array([95, 66, 141], np.uint8) #HSV
    blue_upper = np.array([113, 150, 205], np.uint8) #HSV
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 




    orange_detected = []
   
    ###########################################################
      
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between imageFrame and mask determines 
    # to detect only that particular color 
    kernel = np.ones((5, 5), "uint8") 
      
    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(image, image,  
                              mask = red_mask) 
      
    # For orange color 
    orange_mask = cv2.dilate(orange_mask, kernel) 
    res_orange = cv2.bitwise_and(image, image, 
                                mask = orange_mask) 
      
    # For white color 
    white_mask = cv2.dilate(white_mask, kernel) 
    res_white = cv2.bitwise_and(image, image, 
                               mask = white_mask) 
    
    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(image, image, 
                               mask = blue_mask) 
   
    # # For green color 
    # green_mask = cv2.dilate(green_mask, kernel) 
    # res_blue = cv2.bitwise_and(image, image, 
    #                            mask = green_mask) 


    # Creating contour to track red color 
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    #MAybe save the data, and when this function is almost done, fiind the median of the two values if any. Or just use the only one
    #regarding the arena
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if area > 6000 and area < 8000: # area of cross is aprox 7000
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            center = (int(rect[0][0]), int(rect[0][1]))
            size = (int(rect[1][0] // 2) + 10, int(rect[1][1] // 2) + 10)
            angle = rect[2] + 45
            
            # Create a rotation matrix
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calculate the end points of the cross lines
            end1 = (int(center[0] + size[0] * np.cos(np.radians(angle))),
                    int(center[1] + size[0] * np.sin(np.radians(angle))))
            end2 = (int(center[0] - size[0] * np.cos(np.radians(angle))),
                    int(center[1] - size[0] * np.sin(np.radians(angle))))
            end3 = (int(center[0] + size[1] * np.cos(np.radians(angle + 90))),
                    int(center[1] + size[1] * np.sin(np.radians(angle + 90))))
            end4 = (int(center[0] - size[1] * np.cos(np.radians(angle + 90))),
                    int(center[1] - size[1] * np.sin(np.radians(angle + 90))))
            
            # Draw the cross
            cv2.line(image, end1, end2, (0, 0, 255), 2)
            cv2.line(image, end3, end4, (0, 0, 255), 2)
            
            cv2.putText(image, "Red Colour", (int(rect[0][0]), int(rect[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
            
            no_go_zones = []
            cross_width = 10
            no_go_zones.append([
                (end1, end2),
                (end3, end4)
            ])
            # no_go_zones.append([
            #     (end3[0] - cross_width, end3[1] - cross_width),
            #     (end3[0] + cross_width, end3[1] + cross_width),
            #     (end4[0] + cross_width, end4[1] + cross_width),
            #     (end4[0] - cross_width, end4[1] - cross_width)
            # ])

            save_no_go_zones(no_go_zones)

 
  
    # Creating contour to track orange color 
    contours, hierarchy = cv2.findContours(orange_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y),  
                                       (x + w, y + h), 
                                       (0, 165, 255), 2)  #color of the rectangle, and 2 is the thickness
            # print(f"(x={x}, y={y}) w={w} h={h} area={area}")

              
            cv2.putText(image, "Orange Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX,  
                        1.0, (0, 165, 255)) 
            orange_detected.append(contour)

    check_point_in_orange_region(contours)

  
    # Creating contour to track white color 
    contours, hierarchy = cv2.findContours(white_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), 
                                       (x + w, y + h), 
                                       (255, 255, 255), 2) 
            # print(f"(White objects: x={x}, y={y}) w={w} h={h} area={area}")
            #If a big white object is detected with size of the egg, draw an ellipse to specify the egg
            if(area > 2000 and area < 4000): #before 2900
                image = egg_draw(image,x,y,w,h,area)

            #If a big white object is detected with size of the car
            if(area > 13000 and area < 22000):
                image = car_draw(image,x,y,w,h,area)

            #(x=638, y=683) w=31 h=80 area=813.5 small white square
             
              
            cv2.putText(image, "White Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255)) 
            


    # Program Termination 


    return image

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
                point_in_orange_region = True
                break  # Exit the loop if the point is found in any contour

        if point_in_orange_region:
            orange_balls.append((px, py))
        else:
            white_balls.append((px, py))


    saveOrange_balls(orange_balls)
    saveWhite_balls(white_balls)


def save_no_go_zones(zones, filename="no_go_zones.json"):
    with open(filename, 'w') as file:
        json.dump(zones, file)


def blurred(image):
        
   # Denoising parameters
   h = 10  # filter strength for luminance component
   hForColorComponents = h  # the same filter strength is often used for color
   templateWindowSize = 7  # must be odd, larger values could remove details
   searchWindowSize = 21  # must be odd, larger values are slower

   # Apply fastNlMeansDenoisingColored
   denoised_image = cv2.fastNlMeansDenoisingColored(
     image, None, h, hForColorComponents, templateWindowSize, searchWindowSize
  )


def CannyEdgeGray(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
   cv2.imshow("gray pic", gray) 
   gray = cv2.bilateralFilter(gray, 11, 17, 17)
   cv2.imshow("gray bilateral", gray) 
 #    cv2.imshow("Gray image", gray) 
   gray = cv2.GaussianBlur(gray, (5, 5), 0)
 #    cv2.imshow("Gaussian Blur", gray) 

#Original
#    edged = cv2.Canny(gray, 35, 125)
#    edged = cv2.Canny(gray,100, 200) #no cross to be seen
#    edged = cv2.Canny(gray,0, 105, apertureSize=5)
   edged = cv2.Canny(gray,0, 105)
   cv2.imshow("Canny edge B/W detection", edged) 

   #cropped_image = edged[240:140, 168:167] # Slicing to crop the image

   # Display the cropped image
#    cv2.imshow("cropped", cropped_image)



