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
from config import *
import random
import imutils
from imutils import paths




def save_thresholds(lower_thresh, upper_thresh, color):
    """Function to save the threshold values in config.py"""

    lower_set = False
    upper_set = False

    # Read the existing content of config.py
    with open('config.py', 'r') as file:
        lines = file.readlines()

    # Update the lines with new threshold values
    updated_lines = []
    for line in lines:
        if line.startswith(f'{color.upper()}_LOWER_THRESHOLD'):
            updated_lines.append(
                f'{color.upper()}_LOWER_THRESHOLD = {lower_thresh}\n')
            upper_set = True
        elif line.startswith(f'{color.upper()}_UPPER_THRESHOLD'):
            updated_lines.append(
                f'{color.upper()}_UPPER_THRESHOLD = {upper_thresh}\n')
            lower_set = True
        else:
            updated_lines.append(line)

    # Add the threshold values if they are not already set
    if not lower_set:
        updated_lines.append(
            f'{color.upper()}_LOWER_THRESHOLD = {lower_thresh}\n')
    if not upper_set:
        updated_lines.append(
            f'{color.upper()}_UPPER_THRESHOLD = {upper_thresh}\n')

    # Write the updated content back to config.py
    with open('config.py', 'w') as file:
        file.writelines(updated_lines)


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
    
    # image = cv2.imread( '/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_59_Pro.jpg')
   # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_39_46_Pro.jpg') 
   # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_38_Pro.jpg') #hvid nej
   # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_58_Pro.jpg') 
    # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/pic50upsidedown.jpg') 
    # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240410_10_31_43_Pro.jpg') 
    # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240410_10_31_07_Pro.jpg') 
    # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240410_10_30_54_Pro.jpg') 
    image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240410_10_31_07_Pro.jpg') 

 
  
   
  

#    image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/pic50egghorizontal.jpg') 
    return image


def arena_draw(image, x, y, w, h, area):
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


##Probably remove this... and try to resolve and get to know the canny edge detection 
def cross_draw(image, x, y, w, h, area):
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

    # print("after recreating the points")
    # The points need to be ordered correctly for minAreaRect to work
    rect_points = cv2.convexHull(rect_points)

    # Find the minimum area rectangle
    min_area_rect = cv2.minAreaRect(rect_points)

    """
    minimum area rectangle=((1100.0, 523.5), (167.0, 168.0), 90.0)
    The first two numbers (1100.0, 523.5), shows the x and y-axis of the central point in the square.
    """
    print(f"minimum area rectangle={min_area_rect}") 

    # Convert the rectangle to box points (four corners)
    box = cv2.boxPoints(min_area_rect)
    print(f"box={box}")
    box = np.int0(box)
    # print(f"box2={box}")



    return box, min_area_rect







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



##Maybe "parse" the egg up in two circles. a small and a big one THIS IS NOT IN USE
def egg_detection(image):
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

    ******minRadius=1 and maxRadius=40: The minimum and maximum radius of the circles to be detected.
    """
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=50, param2=30, minRadius=25, maxRadius=30)

  ##

     # Filter out the circles that correspond to the egg
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]  # x, y center and radius of circle
            
            # Draw the outer circle
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            print(f"The center of the circle is at (x={x}, y={y}) and radius is r={r}") #eggs radius is 27
            ##y axis from 225 to 257

            # Draw the center of the circle
            cv2.circle(image, (x, y), 2, (0, 0, 0), 2)

            ####Detection of egg is not properly configured when is laying in different angles...
            center_coordinates = (x,y+10)
            axesLength = (30,45)
            # axesLength = (34,58)
            angle = 0
            startAngle = 0
            endAngle = 360


            image = cv2.ellipse(image, center_coordinates, axesLength, 
            angle, startAngle, endAngle, (0, 255, 0), 3) 


            # Put text 'Egg' near the detected egg
            cv2.putText(image, 'Egg', (x - r, y - r),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Display the result
    # cv2.imshow('Detected eggs', image)


    # return image


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

    ******minRadius=1 and maxRadius=40: The minimum and maximum radius of the circles to be detected.
    """
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=40,
                               param1=50, param2=30, minRadius=5, maxRadius=20)

  ##

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

    # Display the result
    cv2.imshow('Detected Balls', image)


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



def detect_ball_colors(image):
    #https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/
    #https://colorpicker.me/#ffffff
    # https://colorizer.org/
  # Capturing video through webcam 
 #  webcam = cv2.VideoCapture(0) 
  
 # Start a while loop 
 #while(1): 
      
    # Reading the video from the 
    # webcam in image frames 
    # _, imageFrame = webcam.read() 
  
    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    
    # Set range for red color and  
    # define mask 
    red_lower = np.array([0, 113, 180], np.uint8) #HSV
    red_upper = np.array([9, 255, 255], np.uint8) #HSV
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
  
    # Set range for orange color and  
    # define mask  
    """
    2nd attempt on the picture ending with 46, with both higher and lower values at the same time
    I think this was the best so far?
    """
    ##ORIGINAL
    # orange_lower = np.array([11, 121, 215], np.uint8) #HSV
    # orange_upper = np.array([20, 211, 255], np.uint8) #HSV
    # orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 

    orange_lower = np.array([11, 121, 215], np.uint8) #HSV
    orange_upper = np.array([65, 211, 255], np.uint8) #HSV
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 

    """
    2nd attempt on the picture ending with 58, with both higher and lower values at the same time
    """
    # orange_lower = np.array([11, 121, 215], np.uint8) #HSV
    # orange_upper = np.array([20, 211, 255], np.uint8) #HSV
    # orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 

    """
    Found and calculated by median of many hsv values
    """
    # orange_lower = np.array([15, 157, 255], np.uint8) #HSV
    # orange_upper = np.array([35, 177, 255], np.uint8) #HSV
    # orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 
######################################################
   

  
    # Set range for white color and 
    # define mask 
    # white_lower = np.array([94, 80, 2], np.uint8) 
    """
    This is for the "original" picture (ending with 46)
    """
    # white_lower = np.array([22, 0, 181], np.uint8) #HSV
    # white_upper = np.array([36, 126, 255], np.uint8) #HSV
    # white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 

    """
    Attempt on the picture ending with 59
    """
    # white_lower = np.array([0, 0, 255], np.uint8) #HSV
    # white_upper = np.array([39, 59, 255], np.uint8) #HSV
    # white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 




    """
    Average of the two pictures HSV values
    """
    # white_lower = np.array([11, 0, 218], np.uint8) #HSV
    # white_upper = np.array([38, 93, 255], np.uint8) #HSV
    # white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 


    """
    Attempt on the picture ending with 59, with both higher and lower values at the same time
    BEST SO FAR
    """
    #ORIGINAL
    white_lower = np.array([6, 0, 191], np.uint8) #HSV
    white_upper = np.array([179, 42, 255], np.uint8) #HSV
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 

    #second white for getting the darker shade of white
    # for testing purposes (detects every white ball in 38, and misses a lot in 58)
    # white_lower = np.array([0, 25, 177], np.uint8) #HSV
    # white_upper = np.array([31, 74, 255], np.uint8) #HSV
    # white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 

    # white2_lower = np.array([0, 25, 177], np.uint8) #HSV
    # white2_upper = np.array([31, 74, 255], np.uint8) #HSV
    # white2_mask = cv2.inRange(hsvFrame, white2_lower, white2_upper) 
    
    """
    Attempt on the picture ending with 58, with both higher and lower values at the same time
    """
    # white_lower = np.array([0, 0, 233], np.uint8) #HSV
    # white_upper = np.array([179, 42, 255], np.uint8) #HSV
    # white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 

    #########################################################

    # Set range for blue color and 
    # define mask 
    """
    Attempt on the picture ending with 59
    """
    # blue_lower = np.array([55, 69, 100], np.uint8) #HSV
    # blue_upper = np.array([107, 151, 228], np.uint8) #HSV
    # blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 



    """
    Attempt on the picture ending with 46
    """
    blue_lower = np.array([95, 66, 141], np.uint8) #HSV
    blue_upper = np.array([113, 150, 205], np.uint8) #HSV
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 



    # green_lower = np.array([59, 205, 194], np.uint8) #HSV
    # green_upper = np.array([179, 255, 255], np.uint8) #HSV
    # green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 










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
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y),  
                                       (x + w, y + h),  
                                       (0, 0, 255), 2) 
            print(f"(x={x}, y={y}) w={w} h={h} area={area}") #
            if(area > 8000 and area < 15000):
                box, min_area_rect = cross_draw(image,x,y,w,h,area)
                image = cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
              
            cv2.putText(image, "Red Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255))     
  
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
              
            cv2.putText(image, "Orange Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX,  
                        1.0, (0, 165, 255)) 
  
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
            # print(f"(x={x}, y={y}) w={w} h={h} area={area}")
            #If a big white object is detected with size of the egg, draw an ellipse to specify the egg
            if(area > 2900 and area < 4000):
                image = egg_draw(image,x,y,w,h,area)

            #If a big white object is detected with size of the car
            if(area > 13000 and area < 22000):
                image = car_draw(image,x,y,w,h,area)

            #(x=638, y=683) w=31 h=80 area=813.5 small white square
             
              
            cv2.putText(image, "White Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255)) 
            

    # Creating contour to track blue color 
    contours, hierarchy = cv2.findContours(blue_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), 
                                       (x + w, y + h), 
                                       (255, 85, 0), 2) 
              
            cv2.putText(image, "Blue Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 85, 0)) 
              


    # # Creating contour to track green color 
    # contours, hierarchy = cv2.findContours(green_mask, 
    #                                        cv2.RETR_TREE, 
    #                                        cv2.CHAIN_APPROX_SIMPLE) 
    # for pic, contour in enumerate(contours): 
    #     area = cv2.contourArea(contour) 
    #     if(area > 300): 
    #         x, y, w, h = cv2.boundingRect(contour) 
    #         image = cv2.rectangle(image, (x, y), 
    #                                    (x + w, y + h), 
    #                                    (255, 255, 255), 2) 
              
    #         cv2.putText(image, "Blue Colour", (x, y), 
    #                     cv2.FONT_HERSHEY_SIMPLEX, 
    #                     1.0, (255, 2, 0)) 
    # Program Termination 
    cv2.imshow("Multiple Color Detection in Real-TIme", image) 
    # if cv2.waitKey(10) & 0xFF == ord('q'): 
    #     cap.release() 
    #     cv2.destroyAllWindows() 
    #     break  




def detect_ball_colorsVIDEO():
    # Capturing video through webcam 
    webcam = cv2.VideoCapture(0) 
    
    # Start a while loop 
    while(1): 
        
        # Reading the video from the 
        # webcam in image frames 
        _, image = webcam.read() 
    
        # Convert the imageFrame in  
        # BGR(RGB color space) to  
        # HSV(hue-saturation-value) 
        # color space 
        hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
        
        # Set range for red color and  
        # define mask 
        red_lower = np.array([0, 113, 180], np.uint8) #HSV
        red_upper = np.array([9, 255, 255], np.uint8) #HSV
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 
    
        # Set range for orange color and  
        # define mask  
        orange_lower = np.array([11, 121, 215], np.uint8) #HSV
        orange_upper = np.array([65, 211, 255], np.uint8) #HSV
        orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 
    
        # Set range for white color and 
        # define mask 
        white_lower = np.array([6, 0, 191], np.uint8) #HSV
        white_upper = np.array([179, 42, 255], np.uint8) #HSV
        white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 
    
        # Set range for blue color and 
        # define mask 
        blue_lower = np.array([95, 66, 141], np.uint8) #HSV
        blue_upper = np.array([113, 150, 205], np.uint8) #HSV
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper) 
    
        # Morphological Transform, Dilation 
        # for each color and bitwise_and operator 
        # between imageFrame and mask determines 
        # to detect only that particular color 
        kernel = np.ones((5, 5), "uint8") 
        
        # For red color 
        red_mask = cv2.dilate(red_mask, kernel) 
        res_red = cv2.bitwise_and(image, image, mask = red_mask) 
        
        # For orange color 
        orange_mask = cv2.dilate(orange_mask, kernel) 
        res_orange = cv2.bitwise_and(image, image, mask = orange_mask) 
        
        # For white color 
        white_mask = cv2.dilate(white_mask, kernel) 
        res_white = cv2.bitwise_and(image, image, mask = white_mask) 
        
        # For blue color 
        blue_mask = cv2.dilate(blue_mask, kernel) 
        res_blue = cv2.bitwise_and(image, image, mask = blue_mask) 
        
        # Creating contour to track red color 
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2) 
                cv2.putText(image, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))     
      
        # Creating contour to track orange color 
        contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour) 
            if(area > 300): 
                x, y, w, h = cv2.boundingRect(contour) 
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)  
                cv2.putText(image, "Orange Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255)) 
      
        # Creating contour to track white color 
        contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
        for pic, contour in enumerate(contours): 
            area = cv2.contourArea(contour)
        

        # Program Termination 
        cv2.imshow("Multiple Color Detection in Real-TIme", image) 
        if cv2.waitKey(10) & 0xFF == ord('q'): 
           webcam.release() 
           cv2.destroyAllWindows() 
           break


 
def video():
    # Capturing video through webcam 
    webcam = cv2.VideoCapture(0) 
    
    # Start a while loop 
    while(1): 
        
        # Reading the video from the 
        # webcam in image frames 
        _, image = webcam.read() 

            # Program Termination 
        cv2.imshow("Multiple Color Detection in Real-TIme", image) 
        if cv2.waitKey(10) & 0xFF == ord('q'): 
           webcam.release() 
           cv2.destroyAllWindows() 
           break

    




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

# Save or display the denoised image
#cv2.imwrite('denoised_image.png', denoised_image)
# cv2.imshow('Denoised Image', denoised_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def CannyEdgeGray(image):
   gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
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

