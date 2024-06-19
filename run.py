from matplotlib import pyplot as plt 
from typing import Tuple, Optional

from camera2.cam2 import *
from algorithm.algorithm import *
from algorithm.move_to_targetv4 import *
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

# camera_handler = init()

try:   
    while True:
        camera = Camera2()
        video_path = 3
        camera.calibrate_color("red", video_path)
        camera.start_video_stream(video_path, morph=True, record=False)
        
        #ball = SortByDistance()
        move_to_targetv6()
        # cv2.imshow("LiveV2",image)
        
            
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
    comstop = (0,0,0,0,0)
    publish_controller_data(comstop)        
