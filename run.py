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



################ #for testing with pictures ####################
image = cv2.imread('extra/test\images\image.png') 
# find_corners(image)
image = transform(image)
stored_circles = circle_detection(image)
white_detected, orange_detected, cross = detect_ball_colors(image, stored_circles)
white_balls, orange_balls = match_circles_and_contours(image, orange_detected, white_detected, stored_circles) # Returns list of balls
print(white_balls)
car = find_carv2(image)
sorted_list = SortByDistance(car, white_balls, orange_balls, cross)
ball = sorted_list[0]
simulate(white_balls, orange_balls, cross, car, ball)