from picture.utils import *
# from robot import Robot
import cv2
import numpy as np
#from pyimagesearch import imutils
from skimage import exposure
import numpy as np
import argparse
import imutils


#TRUE
def perspectiveTrans(image):
    #xTL,yTL  
    #xTR,yTR
    #xBR,BR
    #xBL,yBL 
    # input coordinates, they are just hard coded for now (generated in get_corners.py) :3
    # top-left, top-right, bottom-right, bottom-left
    input_corners = np.float32([
        # [ 287,   43],
        # [1704,   43],
        # [1704, 1052],
        # [ 287, 1052]
        ################
        # [ 237,    0],
        # [1751,    0],
        # [1751, 1080],
        # [ 237, 1080]
  
        #########################
        #have used this so far, but this is manually pointed out 
        # [ 352,       15   ],       
        # [1746.9998,   15   ],
        # [1746.9998 , 1038.9999],
        # [ 352,     1038.9999]
        #######################
        #This works when switching height and width, but the photo is flipped
        # [ 352,     1038.9999],
        # [ 352,       15   ],
        # [1746.9998,   15   ],
        # [1746.9998 , 1038.9999]
        ######################
        #cross' xy coordinates
        # [ 1016,  440],
        # [1184,  440],
        # [1184,  607],
        # [ 1016,  607]
        #####################

        ######################
        #no. 1 pic in the list in utils
        # [ 360, 1055],
        # [ 288,   43],
        # [1704,   50],
        # [1687, 1041]
        ########################
        [372, 15],
        [1751, 17],
        [1751, 1045],
        [345, 1016]  
       
    ])

    # Width and height of the new image (O.G. picture is 1920x1080)
    #1466x1046 is handpicked of the image     # image = cv2.imread( '/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_59_Pro.jpg')
    # w=1395 h=1024
    # width = 1024#1395 # 1920 #1600
    # height = 1395# 1024 #1080 #1000
    width = 1417#1395  #1395 # 1920 #1600
    height = 765 #1024 # 1024 #1080 #1000
    correct_corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Get the transformation matrix
    matrix = cv2.getPerspectiveTransform(input_corners, correct_corners)

    # Transform
    transformed = cv2.warpPerspective(image, matrix, (width, height))
    return transformed





def detect_arena(image):
    #https://www.geeksforgeeks.org/multiple-color-detection-in-real-time-using-python-opencv/
    #https://colorpicker.me/#ffffff
    # https://colorizer.org/


    # Convert the imageFrame in  
    # BGR(RGB color space) to  
    # HSV(hue-saturation-value) 
    # color space 
    hsvFrame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) 
    
    # Set range for red color and  
    # define mask 
    red_lower = np.array([0, 113, 180], np.uint8) #HSV
    red_upper = np.array([9, 255, 255], np.uint8) #HSV
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper) 




    ###########################################################
      
    # Morphological Transform, Dilation 
    # for each color and bitwise_and operator 
    # between imageFrame and mask determines 
    # to detect only that particular color 
    kernel = np.ones((5, 5), "uint8") 
      
    # For red color 
    red_mask = cv2.dilate(red_mask, kernel) 
    res_red = cv2.bitwise_and(image, image,  
                              mask = red_mask) 

   


    # Creating contour to track red color 
    contours, hierarchy = cv2.findContours(red_mask, 
                                           cv2.RETR_TREE, 
                                           cv2.CHAIN_APPROX_SIMPLE) 
      
    for pic, contour in enumerate(contours): 
        area = cv2.contourArea(contour) 
        if(area > 100000): 
            x, y, w, h = cv2.boundingRect(contour) 
            image = cv2.rectangle(image, (x, y),  
                                       (x + w, y + h),  
                                       (0, 0, 255), 2) 
            # print(f"(x={x}, y={y}) w={w} h={h} area={area}") #
            print(f"Before drawing the area")                             
            if(area > 1250000 and area < 1460000):
                # box, min_area_rect = square_draw(image,x,y,w,h,area)
                # image = cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
                line_draw(image, x, y, w, h, area)
                image = goal_draw(image, x, y)
                image = perspectiveTransDyn(image,x,y,w,h)
                print(f"after drawing the area")
              
            cv2.putText(image, "Red Colour arena", (x, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, 
                        (0, 0, 255))     
  



 
    # Program Termination 
    #cv2.imshow("After transform in detect arena", image) 
    # if cv2.waitKey(10) & 0xFF == ord('q'): 
    #     cap.release() 
    #     cv2.destroyAllWindows() 
    #     break  

    return image


def perspectiveTransDyn(image,x,y,w,h):
    #xTL,yTL  
    #xTR,yTR
    #xBR,BR
    #xBL,yBL 
    # input coordinates, they are just hard coded for now (generated in get_corners.py) :3
    # top-left, top-right, bottom-right, bottom-left
    input_corners = np.float32([
   
        [x, y],
        [x+w, y],
        [x+w, y+h],
        [x, y+h]  
       
    ])

    # Width and height of the new image (O.G. picture is 1920x1080)
    #1466x1046 is handpicked of the image     # image = cv2.imread( '/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_59_Pro.jpg')
    # w=1395 h=1024
    # width = 1024#1395 # 1920 #1600
    # height = 1395# 1024 #1080 #1000
    print(f"width is: w={w} , height is h={h}\nIn perspective trans dyn")
    width = w #1,536 #1417#1395  #1395 # 1920 #1600
    height = h #864 #765 #1024 # 1024 #1080 #1000
    correct_corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Get the transformation matrix
    matrix = cv2.getPerspectiveTransform(input_corners, correct_corners)

    # Transform
    transformed = cv2.warpPerspective(image, matrix, (width, height))
    return transformed



def perspectiveTransCross(image):

#https://pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/
    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height = 300)
    #xTL,yTL  
    #xTR,yTR
    #xBR,BR
    #xBL,yBL 
    # input coordinates, they are just hard coded for now (generated in get_corners.py) :3
    # top-left, top-right, bottom-right, bottom-left
    input_corners = np.float32([
       
        ######################
        #cross' xy coordinates
        [ 1016,  440],
        [1184,  440],
        [1184,  607],
        [ 1016,  607]
        #####################

       
       
    ])

    # Width and height of the new image (O.G. picture is 1920x1080)
    #1466x1046 is handpicked of the image     # image = cv2.imread( '/home/slothie/CDIO_gruppe_7/ppArena/test/images/WIN_20240403_10_40_59_Pro.jpg')
    # w=1395 h=1024
    # width = 1024#1395 # 1920 #1600
    # height = 1395# 1024 #1080 #1000
    width = 1417#1395  #1395 # 1920 #1600
    height = 765 #1024 # 1024 #1080 #1000
    correct_corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Get the transformation matrix
    matrix = cv2.getPerspectiveTransform(input_corners, correct_corners)

    # Transform
    transformed = cv2.warpPerspective(image, matrix, (width, height))
    return transformed