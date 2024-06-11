from arena import *
from matplotlib import pyplot as plt 
from transform_arena import *

image = getImage()

# cv2.imshow('Image', image)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', gray)
#Look at this link for measuring the distance
#https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/


#calibrateColors(image) #Shown without indicators/names
# denoise = cv2.fastNlMeansDenoisingColored(image, None, 10, 100, 7, 21) 



# denoise = cv2.fastNlMeansDenoisingColored(image, None, 10, 100, 7, 21) 
# circle_detection(denoise)  
# detect_ball_colors(denoise)

#https://pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
# ratio = image.shape[0] / 1024 #resize image height to 765 pixels  (used later)
# orig = image.copy()  #copy of original pic

# image = imutils.resize(image, height = 1024)





#Without image correction (Transform)

def basicDetectofImage(image):
    circle_detection(image)  #THIS IS THE GOOD SHIT
    detect_ball_colors(image)
    # circle_detection(image)  #THIS IS THE GOOD SHIT




#Used for detecting objects in picture, and the colors in the image (with image correction, (Transform))
def getMeSomeBallInfo(image):
    # image = perspectiveTrans(image) #Problems when cutting edges off, making the correct size wxh
    image, circles_info = circle_detection(image)  # THIS IS THE GOOD SHIT
    # image, orange_detected = detect_ball_colors(image)
    image = detect_ball_colors(image)
    # check_point_in_orange_region(px, py, contours)
    # Print stored circles information
    # print("Detected and Stored Circles:")
    # for circle in circles_info:
    #     print(circle)
 
    # Call the function to print the balls
    # print_balls("balls.json")








def wBabyCanny(image):
     # image = perspectiveTrans(image) #cant remember the setup..
    # image = detect_arena(image)  #detects arena dynanamically
    calibrate(image)
    image = transform(image)
    circle_detection(image) 
    image = detect_ball_colors(image)
    #CannyEdgeGray(image)
    #cv2.imshow('New image', image)


#testcrosssearch(image)


#image = generateArenaImage(1920, 1, 9)


##Indicates whether it is a ball, or whatever
#image = paintArenaState(image, state)
# denoised_image = paintArenaState(denoised_image, state)

# arena.robot.printRobotState()

##Print image and denoised image
#cv2.imshow('image', image) #Shown with indicators/names
# cv2.imshow('Denoised Image', denoised_image)



#calibrateColors(image)
#paintArenaState(image, state)




#################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
########################            RUN THE DESIRED FUNCTION             ######################## 
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
#################################################################################################






# def getVideo():
#     #172.29.224.1
#     #172.29.224.1
#     stream_url = 'http://172.29.224.1:8080/'

#     # Capturing video through webcam 
#     webcam = cv2.VideoCapture(stream_url)

#     # Start a while loop 
#     while True:
#         # Reading the video from the webcam in image frames 
#         ret, imageFrame = webcam.read()
        
#         if not ret:
#             print("Failed to capture image")
#             break

#         basicDetectofImage(imageFrame)

#         # Display the resulting frame
#         cv2.imshow("Multiple Color Detection in Real-Time", imageFrame)

#         # Break the loop on 'q' key press
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break

#     # Release the webcam and destroy all OpenCV windows
#     webcam.release()
#     cv2.destroyAllWindows()




# getVideo()

# image = blurred(image)
basicDetectofImage(image)

# detect_ball_colors_testbaby2(image)


# getMeSomeBallInfo(image)

# goal_draw(image)

# wBabyCanny(image)


# image = detect_arena(image)
# calibrateColors2(image)

# cv2.imshow("Multiple Color Detection in Real-TIme", image) 


##Comment when running livefeed from webcam
int
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
