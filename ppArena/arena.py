from utils import *
from robot import Robot
import cv2
import numpy as np

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
    #xTL,yTL    xTR,yTR
    #xBL,yBL     xBR,yBR
    # input coordinates, they are just hard coded for now (generated in get_corners.py) :3
    # top-left, top-right, bottom-right, bottom-left
    input_corners = np.float32([
        [ 352,       15   ],       
        [1746.9998,   15   ],
        [1746.9998 , 1038.9999],
        [ 352,     1038.9999]
        #######################
        #This works when switching height and width, but the photo is flipped
        # [ 352,     1038.9999],
        # [ 352,       15   ],
        # [1746.9998,   15   ],
        # [1746.9998 , 1038.9999]
        ######################
        #cross' xy coordinates
        # [ 930,  431],
        # [1082,  431],
        # [1082,  582],
        # [ 930,  582]
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
    width = 1395  #1395 # 1920 #1600
    height = 1024 # 1024 #1080 #1000
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