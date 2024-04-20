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
        [ 360, 1055],
        [ 288,   43],
        [1704,   50],
        [1687, 1041]
        # [372, 15],
        # [1751, 17],
        # [1751, 1045],
        # [345, 1016]  
       
    ])

    # Width and height of the new image (O.G. picture is 1920x1080)
    width = 1466 # 1920 #1600
    height = 1046 #1080 #1000
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