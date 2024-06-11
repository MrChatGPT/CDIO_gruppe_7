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
import random
import imutils
from imutils import paths
import argparse
from skimage import exposure
import json

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

    save_balls(stored_circles)
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
            no_go_zones.append([
                (end1, end2),
                (end3, end4)
            ])

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

            cv2.putText(image, "White Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255)) 
            
    return image

def check_point_in_orange_region(contours):
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

def rgb_to_hsv(rgb):
    color = np.uint8([[rgb]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

def find_car(image_path, output_path='output_image.jpg', yellow_mask_path='yellow_mask.jpg', green_mask_path='green_mask.jpg', center_weight=25):
    # Read the image
    
    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path))
    
    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Manually adjusted HSV ranges for yellow
    yellow_lower_hsv = np.array([20, 100, 100])
    yellow_upper_hsv = np.array([30, 255, 255])
    
    # Define broader HSV ranges for green, gotten after using the two green_hsv1 and green_hsv2 
    green_lower_hsv = np.array([75, 100, 100])
    green_upper_hsv = np.array([95, 255, 255])

    # Create masks for yellow and green
    yellow_mask = cv2.inRange(hsv, yellow_lower_hsv, yellow_upper_hsv)
    green_mask = cv2.inRange(hsv, green_lower_hsv, green_upper_hsv)
    
    # Save the masks for debugging
    #cv2.imwrite(yellow_mask_path, yellow_mask)
    #cv2.imwrite(green_mask_path, green_mask)
    
    # Find contours for yellow and green regions
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Function to find the centroid of the largest contour
    def find_centroid(contours):
        if len(contours) == 0:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    # Find centroids of the largest yellow and green contours
    yellow_centroid = find_centroid(contours_yellow)
    green_centroid = find_centroid(contours_green)
    
    if yellow_centroid is None or green_centroid is None:
        raise ValueError("Could not find the required yellow or green regions in the image.")
    
    # Calculate the center of the car
    center_x = (yellow_centroid[0] + green_centroid[0]) // 2
    center_y = (yellow_centroid[1] + green_centroid[1]) // 2
    
    # Adjust the center position based on the center_weight
    line_vec_x = green_centroid[0] - yellow_centroid[0]
    line_vec_y = green_centroid[1] - yellow_centroid[1]
    line_length = np.sqrt(line_vec_x ** 2 + line_vec_y ** 2)
    
    unit_vec_x = line_vec_x / line_length
    unit_vec_y = line_vec_y / line_length
    
    adjusted_center_x = center_x + int(center_weight * unit_vec_x)
    adjusted_center_y = center_y + int(center_weight * unit_vec_y)
    
    # Calculate the angle of the car with respect to (0,0)
    angle_rad = math.atan2(-line_vec_y, line_vec_x)  # Invert y to account for image coordinate system
    angle_deg = math.degrees(angle_rad)+90
   
    # Ensure the angle is in the range [0, 360)
    if angle_deg < 0:
        angle_deg += 360
    
    #if we wish the angle to be a rounded integer (ex: 180.7010 = 181, 180.46 = 180):
    angle_deg = int(round(angle_deg))
    
    # Draw the centroids, car center, and direction line on the image for visualization
    cv2.circle(image, yellow_centroid, 5, (0, 255, 255), -1) # Yellow centroid
    cv2.circle(image, green_centroid, 5, (0, 255, 0), -1)   # Green centroid
    cv2.circle(image, (adjusted_center_x, adjusted_center_y), 5, (255, 0, 0), -1) # Car center
    cv2.line(image, green_centroid, yellow_centroid, (255, 0, 0), 2) # Direction line
    
    # Save the result
    cv2.imwrite(os.path.join(os.path.dirname(__file__), output_path), image)
    
    # Write the results to a JSON file
    data = [[adjusted_center_x, adjusted_center_y, angle_deg]]
    with open(os.path.join(os.path.dirname(__file__), 'robot.json'), 'w') as json_file:
        json.dump(data, json_file)

    return (adjusted_center_x, adjusted_center_y, angle_deg)

