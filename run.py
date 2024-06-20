import time
import cv2
import numpy as np

from picture.livefeed import CameraHandler
from picture.transform_arena import find_corners, transform
from picture.image_detection import circle_detection, detect_ball_colors, find_carv2, match_circles_and_contours
from algorithm.algorithm import SortByDistance
from algorithm.move_to_target import move_to_target
from algorithm.control import *
import matplotlib.pyplot as plt


def initialize():
    """Initialize the camera handler and calibrate the arena."""
    client.connect()
    
    camera_handler = CameraHandler()
    camera_handler.start_video()
    time.sleep(0.5)
    
    # Calibrate arena transform
    # image = camera_handler._run_video()
    # find_corners(image)
    
    return camera_handler

   

# camera_handler = initialize()

# try:
    
#     while True:
#         image = camera_handler._run_video()
#         image = transform(image)
#         stored_circles = circle_detection(image)
#         white_detected, orange_detected, cross = detect_ball_colors(image, stored_circles)
#         white_balls, orange_balls = match_circles_and_contours(image, orange_detected, white_detected, stored_circles) # Returns list of balls
#         print("White balls: ", white_balls)
#         car = find_carv2(image)
        
#         ball = SortByDistance(car, white_balls, orange_balls, cross)
#         move_to_target(camera_handler, ball)
        
#         plt.figure(figsize=(12.5, 9.5))
#         plt.xlim(0, 1250)
#         plt.ylim(0, 950)
        
#         for wb in white_balls:
#             plt.plot(wb.x, wb.y, 'wo')  # white dot
#         for ob in orange_balls:
#             plt.plot(ob.x, ob.y, 'r-')  # white dot
#         plt.plot((cross.arm[0].start,cross.arm[0].end), (cross.arm[1].start,cross.arm[1].end), 'go')
        
#         plt.plot(car.x, car.y, 'bo')
        
#         plt.show()
# finally:
#     # Ensure the camera is released properly
#     camera_handler.release_camera()
#     comstop = (0, 0, 0, 0, 0)
#     publish_controller_data(comstop)

#for testing with pictures
# def main():
image = cv2.imread('extra/test\images/aaaaa.jpg') 
# find_corners(image)
image = transform(image)

stored_circles = circle_detection(image)
white_detected, orange_detected, cross = detect_ball_colors(image, stored_circles)
white_balls, orange_balls = match_circles_and_contours(image, orange_detected, white_detected, stored_circles) # Returns list of balls
car = find_carv2(image)
ball = SortByDistance(car, white_balls, orange_balls, cross)


plt.figure(figsize=(12.5, 9.5))
plt.xlim(0, 1250)
plt.ylim(950, 0)  # Invert the y-axis to have (0,0) at the top-left corner

for wb in white_balls:
    plt.plot(wb.x, wb.y, 'ko')  # white dot
for ob in orange_balls:
    plt.plot(ob.x, ob.y, 'yo')  # orange dot
for arm in cross.arms:
    plt.plot([arm.start[0], arm.end[0]], [arm.start[1], arm.end[1]], 'go-')
plt.plot(car.x, car.y, 'bo')
plt.plot(ball.x, ball.y, 'mo')
if len(ball.waypoints) < 0:
    print("asica")
    plt.plot(ball.waypoints[0].x, ball.waypoints[0].y, 'mo')
    
car_angle_radians = np.radians(car.angle)
vector_length = 50
vector_end_x = car.x + vector_length * np.cos(car_angle_radians)
vector_end_y = car.y + vector_length * np.sin(car_angle_radians)
plt.arrow(car.x, car.y, vector_end_x - car.x, vector_end_y - car.y, head_width=10, head_length=15, fc='g', ec='g')


plt.show()

