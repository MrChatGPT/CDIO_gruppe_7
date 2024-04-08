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


class Point():
    """Class to represent a point"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


def updateArena(image):
    """Function to update the arena state"""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Blur the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image to get the white elements
    _, whiteBallsThresh = cv2.threshold(
        blurred, WHITE_LOWER_THRESHOLD, WHITE_UPPER_THRESHOLD, cv2.THRESH_BINARY)

    # Threshold the image to get the red elements
    _, redThresh = cv2.threshold(
        blurred, RED_LOWER_THRESHOLD, RED_UPPER_THRESHOLD, cv2.THRESH_BINARY)

    # Get the contours of the white elements
    whiteBallsContours, _ = cv2.findContours(
        whiteBallsThresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the contours of the red elements
    redContours, _ = cv2.findContours(
        redThresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # get red contour area
    redArea = [cv2.contourArea(contour) for contour in redContours]

    # Sort red contours by area
    redContours = [contour for _, contour in sorted(
        zip(redArea, redContours), key=lambda pair: pair[0])]

    # Extract desired red points
    robotHead = getCenter(redContours[0])
    robotTail = getCenter(redContours[1])
    cross = getCenter(redContours[2])

    # get the minimum area rectangle
    minRect = cv2.minAreaRect(redContours[-5])
    box = cv2.boxPoints(minRect)
    # draw the rectangle
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # hget the minimum area rectangle
    minRect = cv2.minAreaRect(redContours[-3])
    box = cv2.boxPoints(minRect)
    # draw the rectangle
    box = np.int0(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    x, y, w, h = cv2.boundingRect(redContours[-2])
    perimeter = [Point(x, y), Point(x, y + h),
                 Point(x + w, y), Point(x + w, y + h)]

    # Get the centers of the white balls
    whiteBallsCenters = [getCenter(contour) for contour in whiteBallsContours]

    return whiteBallsCenters, cross, robotHead, robotTail, perimeter


def getCenter(contour):
    """Function to get the center of a contour from its bounding rectangle"""
    x, y, w, h = cv2.boundingRect(contour)
    return Point(x + w // 2, y + h // 2)


def paintPoint(image, point):
    """Function to print green dots at point centers"""
    cv2.circle(image, (point.getX(), point.getY()), 3, (0, 255, 0), -1)


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



def paintArenaState(image, state):
    """Function to paint the arena state"""

    # Paint the white balls
    for point in state[0]:
        cv2.circle(image, (point.getX(), point.getY()),
                   DOT_RADIUS, (0, 0, 0), -1)
        cv2.putText(image, 'Ball', (point.getX(), point.getY()),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), FONT_SIZE)

    # Paint the cross center
    cv2.circle(image, (state[1].getX(), state[1].getY()),
               DOT_RADIUS, (0, 255, 0), -1)
    cv2.putText(image, 'Cross', (state[1].getX(), state[1].getY(
    )), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), FONT_SIZE)

    # Paint the robot head
    cv2.circle(image, (state[2].getX(), state[2].getY()),
               DOT_RADIUS, (255, 0, 0), -1)
    # Write "head" next to the robot head
    cv2.putText(image, 'Head', (state[2].getX(), state[2].getY()),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), FONT_SIZE)

    # Paint the robot tail
    cv2.circle(image, (state[3].getX(), state[3].getY()),
               DOT_RADIUS, (255, 0, 0), -1)
    # Write "tail" next to the robot tail
    cv2.putText(image, 'Tail', (state[3].getX(), state[3].getY()),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), FONT_SIZE)

    # Paint the perimeter
    for point in state[4]:
        cv2.circle(image, (point.getX(), point.getY()),
                   DOT_RADIUS, (0, 255, 0), -1)
        cv2.putText(image, 'Arena', (point.getX(), point.getY()),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), FONT_SIZE)

    return image

 
     #  '/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_59_Pro.jpg')

      ##threshold has been made from this pic
      #'/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_39_46_Pro.jpg') 
      
def getImage():
    """This is just a dummy function. It will be replaced by the camera module."""
    
    image = cv2.imread( '/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_59_Pro.jpg')
   # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_39_46_Pro.jpg') 
   # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_38_Pro.jpg') #hvid nej
   # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_58_Pro.jpg') 
    # image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/pic50upsidedown.jpg') 

  #  image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/pic50egghorizontal.jpg') 
    return image


def generateArenaImage(res, n_cross, n_balls):
    """This is just a dummy function. It will be replaced by the camera module."""

    hRes = max(res)
    vRes = min(res)
    arenaLength = hRes * 0.8
    arenaWidth = arenaLength*ARENA_RATIO
    arenaRotation = randbelow(15)
    ballDiameter = arenaLength*BALL_RATIO

    cross_size = int(arenaLength*CROSS_RATIO)//2

    # generate a floor
    image = np.zeros((res[1], res[0], 3), np.uint8)
    image[:] = FLOOR_COLOR1

    # generate a rectangle in top of the floor to represent the arena
    perimeter = np.array([[0, 0],
                          [arenaLength, 0],
                          [arenaLength, arenaWidth],
                          [0, arenaWidth]], dtype=np.uint)

    # Generate obstacles
    line1 = [[0, 0-cross_size], [0, 0+cross_size]]
    line2 = [[0-cross_size, 0], [0+cross_size, 0]]
    cross = np.array([line1, line2], dtype=np.int32)

    error = np.random.randint(-arenaWidth*0.1, arenaWidth*0.1, 1)
    cross0 = cross + [arenaLength//2, arenaWidth//2] + error

    error = np.random.randint(-arenaWidth*0.1, arenaWidth*0.1, 1)
    cross1 = cross + [arenaLength//4, arenaWidth//4] + error

    error = np.random.randint(-arenaWidth*0.1, arenaWidth*0.1, 1)
    cross2 = cross + [arenaLength//4, 3*arenaWidth//4] + error

    error = np.random.randint(-arenaWidth*0.1, arenaWidth*0.1, 1)
    cross3 = cross + [3*arenaLength//4, arenaWidth//4] + error

    error = np.random.randint(-arenaWidth*0.1, arenaWidth*0.1, 1)
    cross4 = cross + [3*arenaLength//4, 3*arenaWidth//4] + error

    obstList = np.array([cross0, cross1, cross2, cross3, cross4]).astype(int)

    randoms = np.array(random.sample(list(obstList)[1:], n_cross))

    if n_cross == 1:
        # Ensure it's still an array of the same structure
        # make an array consisting of the romot and the first element of obstList
        obstList = np.array([randoms[0], obstList[0]])

    elif n_cross < 5:
        # Directly assign the result of random.sample without wrapping it into another list
        obstList = np.array([*randoms, obstList[0]])

    # Rotate the obstacles
    for i in range(len(obstList)):
        rotation = random.randint(-20, 20)
        obstList[i] = rotate_structures([obstList[i]], rotation)[0]

    # generate random points
    points = generate_non_overlapping_points(
        n_balls, arenaLength, arenaWidth, ballDiameter, np.concatenate(obstList))

    translation = np.array(
        [(hRes - arenaLength)//2, (vRes - arenaWidth)//2], dtype=np.uint64)

    # translate the points to the center of the arena
    points = np.array(points) + translation

    # rotate the points
    points = rotate_points(points, (hRes//2, vRes//2), arenaRotation)

    # traslate the obstacles to the center of the arena
    for obst in obstList:
        for line in obst:
            line[0] = line[0] + translation
            line[1] = line[1] + translation

    # roate osbtacles
    obstList = rotate_points(obstList, (hRes//2, vRes//2), arenaRotation)

    # translate the perimeter to the center of the arena
    perimeter = perimeter + translation

    # rotate the flattened perimeter
    perimeter = rotate_points(perimeter, (hRes//2, vRes//2), arenaRotation)

    # draw the rectangle
    cv2.polylines(image, [perimeter], True, ARENA_COLOR1, 4)

    # draw the obstacles
    for cross in obstList[1:]:
        for line in cross:
            start_point, end_point = line[0], line[1]
            cv2.line(image, start_point, end_point, ARENA_COLOR1, 4)

    # draw robot
        robot = obstList[0][0]
        head, tail = robot[0], robot[1]

        # addjust the head to tail distance
        head = head + (tail - head)*0.3
        tail = tail + (head - tail)*0.3

        cv2.circle(image, (int(head[0]), int(head[1])), int(
            ballDiameter//2), ARENA_COLOR1, -1)
        cv2.circle(image, (int(tail[0]), int(tail[1])), int(
            ballDiameter), ARENA_COLOR1, -1)

    # draw circles from the points
    for point in points:
        cv2.circle(image, (int(point[0]), int(point[1])), int(
            ballDiameter//2), BALL_COLOR1, -1)

    return image


def showImage(image):
    """This is just a dummy function. It will be replaced by the camera module."""

    cv2.imshow('image', image)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


def calculate_structure_center(structure):
    """Calculate the geometric center of all points in a structure."""
    all_points = np.concatenate(
        structure)  # Flatten the structure into a list of points
    return np.mean(all_points, axis=0)


def rotate_line(line, center, angle_rad):
    """Rotate a line (two points) around a center point."""
    cos_val, sin_val = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_val, -sin_val], [sin_val, cos_val]])

    # Translate line points to origin (subtract center), rotate, then translate back
    rotated_line = np.dot(line - center, rotation_matrix) + center
    return rotated_line


def rotate_structures(structures, angle):
    """
    Rotate each structure (an array of lines) around its center.

    Parameters:
    - structures: An array of structures, each an array of lines, where a line is defined by two points.
    - angle: The rotation angle in degrees.

    Returns:
    - An array of rotated structures.
    """
    angle_rad = np.radians(angle)  # Convert angle from degrees to radians
    rotated_structures = []

    for structure in structures:
        # Find the center of all points in the structure
        center = calculate_structure_center(structure)
        rotated_structure = [rotate_line(
            np.array(line), center, angle_rad) for line in structure]
        rotated_structures.append(rotated_structure)

    return rotated_structures


def rotate_points(points, center, angle):
    """
    Rotate points around a center point.

    Parameters:
    - points: A numpy array of points shaped as (N, 2), where N is the number of points.
    - center: The center point (x, y) for rotation.
    - angle: The rotation angle in degrees.

    Returns:
    - Rotated points as a numpy array shaped as (N, 2).
    """
    shape = points.shape

    # Get the rotation matrix using OpenCV
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Convert points to a shape (N, 1, 2) required by cv2.transform
    points = points.reshape(-1, 1, 2)

    # Apply the rotation
    rotated_points = cv2.transform(points, M)

    # Convert points back to (N, 2)
    return rotated_points.reshape(shape)


def point_line_distance(point, line):
    """Calculate the distance from a point to a line segment more efficiently."""
    p1, p2 = np.array(line[0], dtype=np.float64), np.array(
        line[1], dtype=np.float64)
    p = np.array(point, dtype=np.float64)

    # Line vector
    line_vec = p2 - p1
    # Vector from line's start point to the point
    p_vec = p - p1

    # Project point onto the line (p1p2), yielding the closest point on p1p2 to p
    line_sq_len = np.dot(line_vec, line_vec)
    t = np.dot(p_vec, line_vec) / line_sq_len

    # Clamp t to the range [0, 1] to ensure it lies within the line segment
    t_clamped = np.clip(t, 0.0, 1.0)

    # Calculate the nearest point on the segment to the point
    nearest = p1 + t_clamped * line_vec

    # Calculate the distance from the point to this nearest point on the segment
    dist = np.linalg.norm(p - nearest)

    return dist


def generate_non_overlapping_points(num_points, width, height, min_distance, obstacles):
    """
    Generate a specified number of non-overlapping points within a rectangle.

    Parameters:
    - num_points: The number of points to generate.
    - width, height: Dimensions of the rectangle.
    - min_distance: The minimum allowed distance between any two points.
    - obstacles: A list of obstacles, where each obstacle is a line defined by two points.

    Returns:
    - A list of tuples, where each tuple represents the coordinates of a point.
    """
    points = []
    max_attempts = 10000
    attempts = 0

    # scale min_distance
    min_distance = min_distance * 1.5
    while len(points) < num_points and attempts < max_attempts:
        # Generate a random point within the rectangl

        new_point = np.random.rand(
            1, 2) * [width - min_distance, height - min_distance] + [min_distance/2, min_distance/2]

        # Check if the new point is far enough from all existing points and not overlapping with obstacles
        if (all(np.linalg.norm(new_point - point) >= min_distance for point in points) and
                all(point_line_distance(new_point, obstacle) >= min_distance for obstacle in obstacles[2:, :]) and
                all(point_line_distance(new_point, obstacle) >= min_distance for obstacle in obstacles[:2, :]) and
                all(point_line_distance(new_point, obstacle) >= 3*min_distance for obstacle in obstacles[0:2, :])):
            points.append(new_point[0])
        attempts += 1

    if attempts == max_attempts:
        print("Maximum attempts reached, generated points:", len(points))
    return points


# pik = generateArenaImage(
#     res=(720, 576), n_cross=3, n_balls=50)
# showImage(pik)

def egg_draw(image, x, y, w, h, area):
    # Calculate center coordinates adding 10 to y for visualization purposes
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


##Maybe "parse" the egg up in two circles. a small and a big one
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
            print(f"(x={x}, y={y}) w={w} h={h} area={area}")
            #If a big white object is detected, draw an ellipse to specify the egg
            if(area > 3000 and area < 3800):
                image = egg_draw(image,x,y,w,h,area)
              
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