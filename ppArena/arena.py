from utils import *
from robot import Robot


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
