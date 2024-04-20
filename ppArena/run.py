from arena import *
from matplotlib import pyplot as plt 

image = getImage()

# cv2.imshow('Image', image)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray', gray)
#Look at this link for measuring the distance
#https://pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/

# arena = Arena(image)

# i = 0  # Initialize the variable "i"

# while i < 100:
#     image = getImage()
#     arena.updateState(image)
#     i += 1
#     state = arena.getNewState()

#calibrateColors(image) #Shown without indicators/names
# denoise = cv2.fastNlMeansDenoisingColored(image, None, 10, 100, 7, 21) 



# denoise = cv2.fastNlMeansDenoisingColored(image, None, 10, 100, 7, 21) 
# circle_detection(denoise)  
# detect_ball_colors(denoise)






#Used for detecting circular balls in picture, and the colors in the image
image = perspectiveTrans(image) #Problems when cutting edges off, making the correct size wxh
image, circles_info = circle_detection(image)  #THIS IS THE GOOD SHIT
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

# calibrateColors2(image) #new thres



# detect_ball_colorsVIDEO()

# video()

# CannyEdgeGray(image)




# egg_detection(image) #Noooo.. Not using this...

# imgg = cv2.imread('/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_38_Pro.jpg') 

# gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY) 

# cv2.imshow('gray', gray)
# cv2.imwrite('pic38gray.png', gray)
# imageora = image
# imagewhi = image
#calibrateColors(image) #old thres
# calibrateColors2(imageora)  #new thres
# calibrateColors2(imagewhi)  #new thres

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

int
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
    
#         break
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
