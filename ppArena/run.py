from arena import *
from matplotlib import pyplot as plt 

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






"""
When getting the input corners from the function square_draw, and using the perspective, a big red square is detected... 
has not occured earlier.. 
This has to be solved.
"""

#Without image correction (Transform)

def basicDetectofImage():
    circle_detection(image)  #THIS IS THE GOOD SHIT
    detect_ball_colors(image)




#Used for detecting objects in picture, and the colors in the image (with image correction, (Transform))
def getMeSomeBallInfo():
    # image = perspectiveTrans(image) #Problems when cutting edges off, making the correct size wxh
    image, circles_info = circle_detection(image)  # THIS IS THE GOOD SHIT
    detect_ball_colors(image)

    # Print stored circles information
    print("Detected and Stored Circles:")
    for circle in circles_info:
        print(circle)


"""
(x=550, y=980) w=32 h=42 area=776.5 
and this
{'center': (566, 992), 'radius': 16, 'label': 'Ball'}
is the same orange ball
"""




def wBabyCanny(image):
    image = detect_arena(image)
    # image = perspectiveTrans(image)
    circle_detection(image) 
   # detect_ball_colors(image)
    #CannyEdgeGray(image)
    cv2.imshow('New image', image)


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

# basicDetectofImage()


# goal_draw(image)

wBabyCanny(image)







int
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
