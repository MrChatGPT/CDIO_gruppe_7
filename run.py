from matplotlib import pyplot as plt 
from typing import Tuple, Optional

from picture.livefeed import *
from picture.transform_arena import *
from picture.image_detection import *
from algorithm.algorithm import *
from algorithm.move_to_targetv5 import *
from algorithm.utils import *
from algorithm.control import *




# run image recognition software
# def transform_and_detect(image):
    
    
# initialize program
def init():
    camera_handler = CameraHandler()
    client.connect()

    # Start video in a separate thread
    camera_handler.start_video()
    time.sleep(0.5)
    # for arena transform calibration
    image = camera_handler._run_video()
    
    # find_corners(image)
    return camera_handler

camera_handler = init()

try:   
    while True:
        image = camera_handler._run_video()
        image = transform(image)
        circle_detection(image) 
        image = detect_ball_colors(image)
        image = camera_handler._run_video()
        image = transform(image)
        car = find_carv2(image)
        
        ball = SortByDistance()
        move_to_targetv5(camera_handler, ball)
        # cv2.imshow("LiveV2",image)
        
            
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
    comstop = (0,0,0,0,0)
    publish_controller_data(comstop)        