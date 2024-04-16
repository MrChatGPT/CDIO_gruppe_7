#!/usr/bin/env pybricks-micropython

#https://pybricks.com/ev3-micropython/ev3devices.html#gyroscopic-sensor
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor
from pybricks.parameters import Port, Button, Stop
from pybricks.tools import wait
from pybricks.media.ev3dev import Font
from pybricks.media.ev3dev import ImageFile, Image


import time

from pybricks.tools import print, wait, StopWatch

# Define motors as global variables
motor0 = None
motor1 = None
motor2 = None
motor3 = None

def InitMotors():
    global motor0, motor1, motor2, motor3
    # Initialize a motor at port A. (Medium/Small)
    motor0 = Motor(Port.A)

    # Initialize a motor at port D. (Medium/Small)
    motor3 = Motor(Port.D)

    # Initialize a motor at port B. (LARGE)
    motor1 = Motor(Port.B)

    # Initialize a motor at port C (LARGE)
    motor2 = Motor(Port.C)

####SMALL MOTORS####
def activateSmallMotors(pickupPower):
    global motor0, motor3
    motor0.dc(pickupPower)
    motor3.dc(pickupPower)

def deactivateSmallMotors():
    global motor0, motor3
    motor0.hold()
    motor3.hold()

####LARGE MOTORS####
def deactivateLargeMotors():
    global motor1, motor2
    motor1.hold()
    motor2.hold()

#Functions that take as an argument 'driveTime' about how long to run the motors that are controlling the wheels
def driveStraight(driveTime, drivePower):
    global motor1, motor2
    #Drive forward
    motor1.dc(drivePower)
    motor2.dc(drivePower)
    wait(driveTime)
  
def driveBackwards(driveTime, drivePower):
    global motor1, motor2
    motor1.dc(drivePower)
    motor2.dc(drivePower)
    wait(driveTime)

def driveLeft(driveTime, drivePower):
    global motor1, motor2
    motor1.dc(drivePower) 
    motor2.dc(-drivePower) 
    wait(driveTime)

def driveRight(driveTime, drivePower):
    global motor1, motor2
    motor1.dc(-drivePower) 
    motor2.dc(drivePower) 
    wait(driveTime)
