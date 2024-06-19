from matplotlib import pyplot as plt 
from typing import Tuple, Optional

from camera2.cam2 import *
from algorithm.algorithm import *
from algorithm.move_to_targetv4 import *
from algorithm.utils import *
from algorithm.control import *
from picture.autocalibratecolors import *


    
# camera_handler = init()

try:   
    while True:
        cap, frame
        camera = Camera2(frame)
        move_to_targetv6(camera.waypoint_for_closest_white_ball)
        
        #camera.close
        video_path = 3
        #camera.calibrate_color("red", video_path)
        camera.start_video_stream(video_path, morph=True, record=False)
        camera.frame
        #ball = SortByDistance()
        move_to_targetv6()
        # cv2.imshow("LiveV2",image)
        
            
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
    comstop = (0,0,0,0,0)
    publish_controller_data(comstop)        
