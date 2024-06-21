import time
import cv2
import numpy as np

from picture.livefeed import CameraHandler
from picture.transform_arena import find_corners, transform
from picture.image_detection import circle_detection, detect_ball_colors, find_carv2, match_circles_and_contours
from algorithm.algorithm import SortByDistance
from algorithm.move_to_target import move_to_target
from algorithm.control import *
from simulation import simulate

class Ball:
    def __init__(self, x, y, obstacle=0):
        self.x = x
        self.y = y
        self.obstacle = obstacle
        self.waypoints = []

    def add_waypoint(self, waypoint):
        self.waypoints.append(waypoint)

    def clear_waypoints(self):
        self.waypoints = []
        
    def pop_waypoint(self):
        return self.waypoints.pop()

    def __repr__(self):
        return f"Ball(x={self.x}, y={self.y}, obstacle={self.obstacle}, waypoints={self.waypoints})"

def initialize():
    """Initialize the camera handler and calibrate the arena."""
    client.connect()
    
    camera_handler = CameraHandler()
    camera_handler.start_video()
    time.sleep(0.5)
    
    # Calibrate arena transform
    image = camera_handler._run_video()
    find_corners(image)
    
    return camera_handler

   

# camera_handler = initialize()

# try:
    
#     while True:
#         image = camera_handler._run_video()
#         image = transform(image)
#         stored_circles = circle_detection(image)
#         white_detected, orange_detected, cross = detect_ball_colors(image, stored_circles)
#         white_balls, orange_balls = match_circles_and_contours(image, orange_detected, white_detected, stored_circles) # Returns list of balls
#         car = find_carv2(image)
        
#         ball = SortByDistance(car, white_balls, orange_balls, cross)
#         move_to_target(camera_handler, ball)
        
#         
# finally:
#     # Ensure the camera is released properly
#     camera_handler.release_camera()
#     comstop = (0, 0, 0, 0, 0)
#     publish_controller_data(comstop)



################ #for testing with pictures ####################
image = cv2.imread('extra/test\images\image.png') 
# find_corners(image)
image = transform(image)
stored_circles = circle_detection(image)
white_detected, orange_detected, cross = detect_ball_colors(image, stored_circles)
white_balls, orange_balls = match_circles_and_contours(image, orange_detected, white_detected, stored_circles) # Returns list of balls
print(cross.angle)
white_balls = []
car = find_carv2(image)
sorted_list = SortByDistance(car, white_balls, orange_balls, cross)
ball = sorted_list[0]
print(ball)
simulate(white_balls, orange_balls, cross, car, ball)