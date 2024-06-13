from matplotlib import pyplot as plt 
from typing import Tuple, Optional

from picture.livefeed import *
from picture.transform_arena import *
from picture.image_detection import *
from algorithm.algorithm import *
from algorithm.move_to_target import *
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

    # for arena transform calibration
    image = camera_handler._run_video()
    find_corners(image)
    return camera_handler

camera_handler = init()

# run program
# try:
#     image = camera_handler._run_video()
#     image = transform(image)
#     circle_detection(image) 
#     image = detect_ball_colors(image)
#     while True:
#         image = camera_handler._run_video()
#         image = transform(image)
        
#         car = find_car(image)
#         #---- her forventes der at være 4 json filer: -------
#         # nogo_zones, ping_pong_balls, orange_ball og robot
#         # ---------------------------------------------------
#         # Herfra mangles: 
#         # algoritme som henter data fra .json filerne og så bevæger bilen
#         move_to_target(SortByDistance())
        
        
# finally:
#     # Ensure the camera is released properly
#     camera_handler.release_camera()
#     comstop = (0,0,0,0,0)
#     publish_controller_data(comstop)
    
try:   
    while True:
        check = 0
        print(f"Check value (0 for new ball positions, else 1): {check}")
        image = camera_handler._run_video()
        image = transform(image)
        circle_detection(image) 
        image = detect_ball_colors(image)
        while not check:
            image = camera_handler._run_video()
            image = transform(image)
            
            car = find_carv2(image)
            #---- her forventes der at være 4 json filer: -------
            # nogo_zones, ping_pong_balls, orange_ball og robot
            # ---------------------------------------------------
            # Herfra mangles: 
            # algoritme som henter data fra .json filerne og så bevæger bilen
            check = move_to_target(SortByDistance())       
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
    comstop = (0,0,0,0,0)
    publish_controller_data(comstop)        
