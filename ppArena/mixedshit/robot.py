from picture.utils import *


class Robot():
    """Class to represent the robot"""

    def __init__(self, arenaState):
        self.newHead = arenaState[2]
        self.newTail = arenaState[3]
        self.oldHead = self.newHead
        self.oldTail = self.newTail
        self.speed = 0
        self.magazine_count = 0

    def getHead(self):
        return self.newHead

    def getTail(self):
        return self.newTail

    def setHead(self, head):
        self.oldHead = self.newHead
        self.newHeadhead = head

    def setTail(self, tail):
        self.oldTail = self.newTail
        self.newTail = tail

    def getSpeed(self):
        return self.speed

    def getMagazineCount(self):
        return self.magazine_count

    def pickUpBall(self):
        # TODO: Implement this
        self.magazine_count += 1

    def dropBalls(self):
        # TODO: Implement this
        self.magazine_count = 0

    def drive(self, duration, direction):
        """Function to drive the robot"""
        if direction == 1:
            # TODO: Implement this (drive forward for duration seconds)
            pass
        else:
            # TODO: Implement this (drive backward for duration seconds)
            pass

    def turn(self, duration, direction):
        if direction == 1:
            # TODO: Implement this (turn left for duration seconds)
            pass
        else:
            # TODO: Implement this (turn right for duration seconds)
            pass

    def printRobotState(self):
        print('Robot state:')
        print('Head: ({}, {})'.format(self.newHead.getX(), self.newHead.getY()))
        print('Tail: ({}, {})'.format(self.newTail.getX(), self.newTail.getY()))
        print('Speed: {}'.format(self.speed))
        print('Magazine count: {}'.format(self.magazine_count))
