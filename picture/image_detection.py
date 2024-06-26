import math
import os
import cv2
from cv2 import GaussianBlur
import numpy as np
import random
import imutils
import argparse
import json



# Function to check if a point is within any detected orange region
def check_point_in_orange_region(contours):
    # print_balls("balls.json")
    
    # #To store balls in separate arrays
    # white_balls = []
    # orange_balls = []

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
    #     # if point_in_orange_region:
    #     #     print(f"The point ({px}, {py}) is within an orange region.")
    #     #     orange_balls.append((px, py))
    #     # else:
    #     #     print(f"The point ({px}, {py}) is not within any orange region.")
    #     #     white_balls.append((px, py))

    # saveOrange_balls(orange_balls)
    # saveWhite_balls(white_balls)

def egg_draw(image, x, y, w, h, area):
    # Load the ball coordinates
    balls = load_balls("balls.json")
    
    # Check if any ball coordinates are within the given area
    if is_ball_in_area(balls, x, y, w, h):
        return image  # Skip drawing the ellipse if a ball is found in the area

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
    # Create the dictionary
    egg = {
    "center_coordinates": center_coordinates,
    "axesLength": axesLength,
    "angle": angle,
    "startAngle": startAngle,
    "endAngle": endAngle
    }
    save_Egg(egg)
    
    # Put text 'Egg' near the detected egg
    cv2.putText(image, 'Egg', (center_coordinates[0] - 10, center_coordinates[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    return image

def is_ball_in_area(balls, x, y, w, h):
    for (bx, by) in balls:
        if x <= bx <= x + w and y <= by <= y + h:
            return True
    return False


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
    red_lower = np.array([0, 113, 180], np.uint8) #HSV   0, 113, 180 # 6, 128, 244
    red_upper = np.array([9, 255, 255], np.uint8) #HSV  9, 255, 255 # 10, 163, 255
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    orange_lower = np.array([11, 121, 215], np.uint8) #HSV
    orange_upper = np.array([65, 230, 255], np.uint8) #HSV 65, 211, 255
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

    # Set range for pink color and  
    # define mask 
    pink_lower = np.array([154,  62,  80], np.uint8) #HSV 
    pink_upper = np.array([169, 105, 255], np.uint8) #HSV  
    pink_mask = cv2.inRange(hsvFrame, pink_lower, pink_upper) 

    # Set range for green color and  
    # define mask 
    green_lower = np.array([72, 130, 187], np.uint8) #HSV   51,  87, 182
    green_upper = np.array([129, 241, 255], np.uint8) #HSV   89, 255 , 255
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 



    yellow_lower = np.array([20, 131, 199], np.uint8) #HSV  28,  82, 247   # 20,  40, 247 # 20, 131, 199
    yellow_upper = np.array([ 50, 202, 255], np.uint8) #HSV 46, 172, 255   #  27, 202, 255
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper) 

    orange_detected = []
    # point_in_orange_region = False
    #px, py = 1302, 166
   

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
      

    
    # For blue color 
    blue_mask = cv2.dilate(blue_mask, kernel) 
    res_blue = cv2.bitwise_and(image, image, 
                               mask = blue_mask) 
   
    # For pink color 
    pink_mask = cv2.dilate(pink_mask, kernel) 
    res_pink = cv2.bitwise_and(image, image,  
                              mask = pink_mask) 
    # # For green color 
    green_mask = cv2.dilate(green_mask, kernel) 
    res_green = cv2.bitwise_and(image, image, 
                               mask = green_mask) 
    
    # # For yellow color 
    yellow_mask = cv2.dilate(yellow_mask, kernel) 
    res_yellow = cv2.bitwise_and(image, image, 
                               mask = yellow_mask) 
    


    # Morphological Transform, Erosion followed by Dilation
    # kernel = np.ones((9, 9), "uint8")
    kernel = np.ones((6, 6), "uint8")
    white_mask = cv2.erode(white_mask, kernel, iterations=1)
    white_mask = cv2.dilate(white_mask, kernel, iterations=2)

 




    # Creating contour to track red color 
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    #MAybe save the data, and when this function is almost done, fiind the median of the two values if any. Or just use the only one
    #regarding the arena
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if area > 5000: 
            x, y, w, h = cv2.boundingRect(contour) 
            if area > 6000 and area < 8000:  # area of cross is aprox 7000
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
                
                no_go_zones = [
                    (end1, end2),
                    (end3, end4),
                    (center, angle)
                ]
                
                save_no_go_zones(no_go_zones)

                    
                
                cv2.putText(image, "Red Colour utils", (x, y), 
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
            # image = cv2.rectangle(image, (x, y),  
            #                            (x + w, y + h), 
            #                            (0, 165, 255), 2)  #color of the rectangle, and 2 is the thickness
            orange_detected.append(contour)
            # check_point_in_orange_region(contours)
              
            cv2.putText(image, "Orange Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX,  
                        1.0, (0, 165, 255)) 
            # orange_detected.append(contour)
    #check_point_in_orange_region(contours)
    image, matched_circles = match_circles_and_contours(image, orange_detected)
    check_point_in_orange_region(orange_detected)
   

    


    # Creating contour to track white color 
    # Additional preprocessing to separate close objects
    blurred = cv2.GaussianBlur(white_mask, (5, 5), 0) 
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Use distance transform and watershed algorithm to separate objects
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)  #0.5*
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(thresh, sure_fg)
    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image, markers)
    # image[markers == -1] = [0, 0, 255]


    contours, hierarchy = cv2.findContours(white_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 450): 
            x, y, w, h = cv2.boundingRect(contour) 
            # image = cv2.rectangle(image, (x, y), 
            #                            (x + w, y + h), 
            #                            (255, 255, 255), 2) 
            # print(f"(White objects: x={x}, y={y}) w={w} h={h} area={area}")
            #If a big white object is detected with size of the egg, draw an ellipse to specify the egg
            if(area > 2000 and area < 4000): #before 2900
                image = egg_draw(image,x,y,w,h,area)
            # #If a big white object is detected with size of the car
            # if(area > 13000 and area < 22000):
            #     image = car_draw(image,x,y,w,h,area)
            #(x=638, y=683) w=31 h=80 area=813.5 small white square
             
              
            cv2.putText(image, "White Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (255, 255, 255)) 
            
    



    # # Creating contour to track green color 
    contours, hierarchy = cv2.findContours(green_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 500): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), 
                                       (x + w, y + h), 
                                       (0, 255, 0), 2) 
            cv2.putText(image, "Green Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 255, 0)) 
            
    # # Creating contour to track yellow color 
    contours, hierarchy = cv2.findContours(yellow_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 800): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y), 
                                       (x + w, y + h), 
                                       (0, 0, 0), 2) 
            # line_drawForPat(image, x, y, w, h, area)
            cv2.putText(image, "yellow Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.0, (0, 0, 0)) 




    # Program Termination 
    # cv2.imshow("Multiple Color Detection in Real-TIme utils", image) 

    # return orange_detected, image
    return image
    # if cv2.waitKey(10) & 0xFF == ord('q'): 
    #     cap.release() 
    #     cv2.destroyAllWindows() 
    #     break  




def match_circles_and_contours(image, contours):

    #To store balls in separate arrays
    white_balls = []
    orange_balls = []

    # Check each ball coordinate
    balls = load_balls("balls.json")
    matched_circles = []
    for cx, cy in balls:
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= cx <= x + w and y <= cy <= y + h:
                matched_circles.append((cx, cy))
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(image, "Orange Colour", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255))
                orange_balls.append((cx, cy))
            else:
                white_balls.append((cx, cy))

            
    
    
    saveOrange_balls(orange_balls)
    saveWhite_balls(white_balls)


    return image, matched_circles




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

def save_Egg(egg, filename="egg.json"):
    with open(filename, 'w') as file:
        json.dump(egg, file, indent=4)

def save_no_go_zones(zones, filename="no_go_zones.json"):
    with open(filename, 'w') as file:
        json.dump(zones, file)



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

def rgb_to_hsv(rgb):
    color = np.uint8([[rgb]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

def find_car(image, output_path='output_image.jpg', yellow_mask_path='yellow_mask.jpg', green_mask_path='green_mask.jpg', center_weight=25):
    # Read the image
    
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
        print("car is goone")
        return
        
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
    with open('robot.json', 'w') as json_file:
        json.dump(data, json_file)
    
    return (adjusted_center_x, adjusted_center_y, angle_deg)

