from utils import *
from robot import Robot
import cv2
import numpy as np
#from pyimagesearch import imutils
from skimage import exposure
import numpy as np
import argparse
import imutils


class Arena():
    """Class to represent the arena"""

    def __init__(self, image):
        self.oldState = updateArena(image)
        self.newState = self.oldState
        self.robot = Robot(self.newState)

    def getOldState(self):
        return self.oldState

    def getNewState(self):
        return self.newState

    def updateState(self, image):
        """Function to update the arena state"""
        self.oldState = self.newState
        self.newState = updateArena(image)
        self.robot.setHead(self.newState[2])
        self.robot.setTail(self.newState[3])

    def planAction(self):
        """Function to plan the next action"""
        # TODO: Implement this
        #   Drive to goal

        pass

    def executeAction(self):
        """Function to execute the next action"""
        # TODO: Implement this
        pass


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
        [ 1016,  440],
        [1184,  440],
        [1184,  607],
        [ 1016,  607]
        #####################

        ######################
        #no. 1 pic in the list in utils
        # [ 360, 1055],
        # [ 288,   43],
        # [1704,   50],
        # [1687, 1041]
        ########################
        # [372, 15],
        # [1751, 17],
        # [1751, 1045],
        # [345, 1016]  
       
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