from matplotlib import pyplot as plt 
from picture.livefeed import *
from picture.transform_arena import *
from picture.image_detection import *
# from algorithm.algorithm import *
# from algorithm.move_to_target import *
from extra.color_calibration import *



# run image recognition software
def transform_and_detect(image):
    # image = transform(image)
    circle_detection(image) 
    image = detect_ball_colors(image)
    car = find_car(image,center_weight=150)
    cv2.imshow("Picture", image)
    


# initialize program
def init():
    camera_handler = CameraHandler()

    # Start video in a separate thread
    camera_handler.start_video()

    # for arena transform calibration
    image = camera_handler._run_video()
    #find_corners(image)
    return camera_handler

# camera_handler = init()

# run program
# try:
#     while True:
#         image = camera_handler._run_video()
#         transform_and_detect(image)
#         #---- her forventes der at være 4 json filer: -------
#         # nogo_zones, ping_pong_balls, orange_ball og robot
#         # ---------------------------------------------------
#         # Herfra mangles: 
#         # algoritme som henter data fra .json filerne og så bevæger bilen
#         # move_to_target(SortByDistance())
        
        
# finally:
#     # Ensure the camera is released properly
#     camera_handler.release_camera()

# image = cv2.imread('/home/slothie/fullbranch/CDIO_gruppe_7/extra/test/images/WIN_20240610_14_26_12_Pro.jpg')
# image = cv2.imread('/home/slothie/fullbranch/CDIO_gruppe_7/extra/test/images/WIN_20240610_15_02_15_Pro.jpg')
image = cv2.imread('extra/test/images/WIN_20240614_09_46_53_Pro.jpg')
# /home/slothie/fullbranch/CDIO_gruppe_7/extra/test/images/WIN_20240610_14_26_12_Pro.jpg
transform_and_detect(image)
#calibrateColors2(image)

try:
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
