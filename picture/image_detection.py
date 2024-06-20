import math
import os
import cv2
from cv2 import GaussianBlur
import numpy as np
import random
import imutils
import argparse
import json

class Ball:
    def __init__(self, x, y, obstacle=0):
        self.x = x
        self.y = y
        self.obstacle = obstacle
        self.waypoints = []

    def add_waypoint(self, waypoint):
        print(f"Waypoint added at: {waypoint}, on {self}")
        self.waypoints.append(waypoint)

    def clear_waypoints(self):
        self.waypoints = []
        
    def pop_waypoint(self):
        return self.waypoints.pop()

    def __repr__(self):
        return f"Ball(x={self.x}, y={self.y}, obstacle={self.obstacle}, waypoints={self.waypoints})"

class Cross:
    class Arm:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def __init__(self, x, y, angle, arms):
        self.x = x
        self.y = y
        self.angle = angle
        self.arms = [self.Arm(*arm) for arm in arms]

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle

    def __repr__(self):
        return f"Car(x={self.x}, y={self.y}, angle={self.angle})"

# Function to check if a point is within any detected orange region
def check_point_in_orange_region(contours):
    
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

    center_coordinates = (x + w//2, y + h//2)
    axesLength = (w//2, h//2)
    angle = 0
    startAngle = 0
    endAngle = 360
   
    egg = {
    "center_coordinates": center_coordinates,
    "axesLength": axesLength,
    "angle": angle,
    "startAngle": startAngle,
    "endAngle": endAngle
    }
    
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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)  
    
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=18, 
                               param1=50, param2=20, minRadius=12, maxRadius=17) #minRadius=5, maxRadius=20 , param2= 28 ORIG
  
    stored_circles = []
     # Filter out the circles that correspond to the ping pong balls
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]  # x, y center and radius of circle
            
            stored_circles.append((x,y))
    return stored_circles
    


def detect_ball_colors(image, stored_circles):
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    ############ Thresholds #####################
    red_lower = np.array([0, 113, 180], np.uint8) #HSV_old  0, 113, 180
    red_upper = np.array([9, 255, 255], np.uint8) #HSV_old  9, 255, 255
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 

    orange_lower = np.array([13, 135, 223], np.uint8) #HSV_old 5, 156, 184
    orange_upper = np.array([44, 255, 255], np.uint8) #HSV_old 50, 255, 255
    orange_mask = cv2.inRange(hsvFrame, orange_lower, orange_upper) 

    white_lower = np.array([0, 0, 243], np.uint8) #HSV_old 15, 0, 200
    white_upper = np.array([203, 61, 255], np.uint8) #HSV_old 41, 59, 255
    white_mask = cv2.inRange(hsvFrame, white_lower, white_upper) 
    
    green_lower = np.array([44, 56, 141], np.uint8) #HSV_old   72, 130, 187
    green_upper = np.array([179, 255, 255], np.uint8) #HSV_old   129, 241, 255
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper) 

    orange_detected = []
    white_detected = []
    
    ##################### Create masks #################################
    kernel = np.ones((5, 5), "uint8") 
      
    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
          
    # For orange color 
    orange_mask = cv2.dilate(orange_mask, kernel) 
     
    # # For green color 
    green_mask = cv2.dilate(green_mask, kernel) 
    
    # Morphological Transform, Erosion followed by Dilation
    # kernel = np.ones((9, 9), "uint8")
    kernel = np.ones((6, 6), "uint8")
    white_mask = cv2.erode(white_mask, kernel, iterations=1)
    white_mask = cv2.dilate(white_mask, kernel, iterations=2)

    # Creating contour to track red color 
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE)
      
    ######################### Cross detection ##########################
    [[[752, 344], [741, 209]], [[679, 282], [814, 271]], [[747, 277], 85.38935089111328]]
    cross = Cross(0,0,0,[(0,0)])
    for pic,contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        
        if area > 3000: 
            x, y, w, h = cv2.boundingRect(contour) 
            if area > 3000 and area < 8000:  # area of cross is aprox 7000
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
                
                # DEBUG: Draw the cross
                # cv2.line(image, end1, end2, (0, 0, 255), 2)
                # cv2.line(image, end3, end4, (0, 0, 255), 2)
                cross = Cross(center[0],center[1], angle, [(end1, end2), (end3, end4)])
    
    if cross.x == 0:
        print("no cross detected")
 
    ################## Detect orange balls ######################
    contours, hierarchy = cv2.findContours(orange_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 300):
            x, y, w, h = cv2.boundingRect(contour) 

            orange_detected.append(contour)
          
   
    ##################### Detect white balls and egg ###########################
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

    contours, hierarchy = cv2.findContours(white_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 450): 
            x, y, w, h = cv2.boundingRect(contour) 
            white_detected.append(contour)
           
            if(area > 2000 and area < 4000): #before 2900
                center_coordinates = (x + w//2, y + h//2)
                axesLength = (w//2, h//2)
                angle = 0
                startAngle = 0
                endAngle = 360

                egg = {
                "center_coordinates": center_coordinates,
                "axesLength": axesLength,
                "angle": angle,
                "startAngle": startAngle,
                "endAngle": endAngle
                }
                save_Egg(egg) # I dont know if we need this
              
    return white_detected, orange_detected, cross



def match_circles_and_contours(image, orange_detected, white_detected, stored_circles):

    #To store balls in separate arrays
    white_balls = []
    orange_balls = []

    # Check each ball coordinate
    balls = stored_circles
    for cx, cy in balls:
        for contour in orange_detected:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= cx <= x + w and y <= cy <= y + h:
                orange_balls.append(Ball(cx, cy))
        
        for contour in white_detected:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= cx <= x + w and y <= cy <= y + h:
                white_balls.append(Ball(cx, cy))

    # saveOrange_balls(orange_balls)
    # saveWhite_balls(white_balls)


    return white_balls, orange_balls

            
    
    
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

def save_no_go_zones(zones, filename="no_go_zones.json"):
    with open(filename, 'w') as file:
        json.dump(zones, file)

def rgb_to_hsv(rgb):
    color = np.uint8([[rgb]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

def find_carv2(image, output_image_path='output_image.jpg'):
    # Read the mask image
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green_lower = np.array([44, 56, 141], np.uint8)
    green_upper = np.array([179, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Separate contours into front (squares) and back (rectangle)
    front_contours = []
    back_contour = None
    
    # Filter out very small contours (noise)
    min_contour_area = 200  # Adjust this threshold as needed
    valid_contours = [contour for contour in contours if cv2.contourArea(contour) > min_contour_area]
    for contour in valid_contours:
        area = cv2.contourArea(contour)
        if area > 4000:
            back_contour = contour
        else:
            front_contours.append(contour)
    
    # Ensure we have exactly one back contour and two front contours
    if len(front_contours) != 2:
        print("car is gone")
        return
    
    # Calculate the bounding box for all contours
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')
    for contour in valid_contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)
    
    # Calculate the center of the bounding box
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    
    # Calculate the centroid of the back (rectangle)
    x, y, w, h = cv2.boundingRect(back_contour)
    back_x = x + w // 2
    back_y = y + h // 2
    
    # Calculate the centroid of the front (average of two squares)
    front_x = 0
    front_y = 0
    for contour in front_contours:
        x, y, w, h = cv2.boundingRect(contour)
        front_x += x + w // 2
        front_y += y + h // 2
    front_x //= 2
    front_y //= 2



    
    # Calculate the angle
    angle_rad = math.atan2(front_y - center_y, front_x - center_x)
    
    angle_deg = math.degrees(angle_rad)

    angle_deg = (angle_deg)%360 

    car = Car(center_x, center_y, angle_deg)
    
    # DEBUG
    # Draw the centroids, car center, and direction arrow on the image for visualization
    # cv2.circle(image, (back_x, back_y), 5, (0, 0, 255), -1)  # Back centroid (red)
    # cv2.circle(image, (front_x, front_y), 5, (0, 255, 0), -1)  # Front centroid (green)
    # cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # Car center (blue)
    # cv2.arrowedLine(image, (back_x, back_y), (front_x, front_y), (255, 0, 0), 2)  # Direction arrow (blue)
    # cv2.imwrite(output_image_path, image)

    # Save the data to robot.json
    # data = [[center_x, center_y, angle_deg]]
    # with open('robot.json', 'w') as json_file:
    #     json.dump(data, json_file)
    return car