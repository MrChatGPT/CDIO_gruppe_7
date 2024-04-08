

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
    image = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/pic50upsidedown.jpg') 

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
