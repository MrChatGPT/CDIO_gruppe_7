import cv2
import numpy as np
from cv2 import GaussianBlur
from utils import *
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


# image = getImage()
# image= cv2.imread('newcar/WIN_20240610_15_04_20_Pro.jpg') #miss 1 w NOB

# Function to check if a point is within any detected orange region
def check_point_in_orange_region(contours):
    white_balls = []
    orange_balls = []

    balls = load_balls("balls.json")
    for px, py in balls:
        point_in_orange_region = False
        for contour in contours:
            dist = cv2.pointPolygonTest(contour, (px, py), False)
            if dist >= 0:
                point_in_orange_region = True
                break

        if point_in_orange_region:
            print(f"The point ({px}, {py}) is within an orange region.")
            orange_balls.append((px, py))
        else:
            print(f"The point ({px}, {py}) is not within any orange region.")
            white_balls.append((px, py))

    saveOrange_balls(orange_balls)
    saveWhite_balls(white_balls)

def circle_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=18, 
                               param1=50, param2=24, minRadius=12, maxRadius=17)

    stored_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            x, y, r = i[0], i[1], i[2]
            cv2.circle(image, (x, y), r, (0, 255, 0), 1)
            cv2.circle(image, (x, y), 2, (0, 0, 0), 2)
            stored_circles.append({'center': (x, y), 'radius': r, 'label': 'Ball'})
            save_balls(stored_circles)

    return image, stored_circles

def find_oranges(image, contours):
    orange_detected = []
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour)
        if area > 300:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"(Orange x={x}, y={y}) w={w} h={h} area={area}")
            orange_detected.append(contour)
            cv2.putText(image, "Orange Colour", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255))
    return orange_detected

def match_circles_and_contours(image, circles, contours):
    matched_circles = []
    for circle in circles:
        cx, cy, radius = circle['center'][0], circle['center'][1], circle['radius']
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if x <= cx <= x + w and y <= cy <= y + h:
                matched_circles.append(circle)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 165, 255), 2)
                cv2.putText(image, "Orange Colour", (x, y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255))
                break
    return image, matched_circles

# Load image and contours
# image = cv2.imread('/mnt/data/image.png')
image= cv2.imread('newcar/WIN_20240610_15_04_20_Pro.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

orange_detected = find_oranges(image, contours)
image, stored_circles = circle_detection(image)
image, matched_circles = match_circles_and_contours(image, stored_circles, orange_detected)
check_point_in_orange_region(orange_detected)

cv2.imshow('Detected Oranges and Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
