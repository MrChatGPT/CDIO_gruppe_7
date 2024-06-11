from matplotlib import pyplot as plt 
from picture.livefeed import *
from picture.transform_arena import *
from picture.image_detection import *

# run image recognition software
def transform_and_detect(image):
    image = transform(image)
    circle_detection(image) 
    image = detect_ball_colors(image)

# initialize program
def init():
    camera_handler = CameraHandler()

    # Start video in a separate thread
    camera_handler.start_video()

    # for arena transform calibration
    image = camera_handler._run_video()
    find_corners(image)
    return camera_handler

camera_handler = init()

# run program
try:
    while True:
        image = camera_handler._run_video()
        transform_and_detect(image)
        
finally:
    # Ensure the camera is released properly
    camera_handler.release_camera()
