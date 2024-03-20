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

def InitMotors():
  
 # Initialize a motor at port A. (Medium/Small)
 motor0 = Motor(Port.A)

 # Initialize a motor at port D. (Medium/Small)
 motor3 = Motor(Port.D)

 # Initialize a motor at port B. (LARGE)
 motor1 = Motor(Port.B)

 # Initialize a motor at port C (LARGE)
 motor2 = Motor(Port.C)



####SMALL MOTORS####
def activateSmallMotors():
  motor0.dc(30)
  motor3.dc(30)

def deactivateSmallMotors():
  motor0.hold()
  motor3.hold()

####LARGE MOTORS####
def deactivateLargeMotors():
  motor1.hold()
  motor2.hold()



#Functions that take as an argument 'driveTime' about how long to run the motors that are controlling the wheels
def driveStraight(driveTime):
  #Drive forward
  motor1.dc(20)
  motor2.dc(20)
  wait(driveTime)
  

def driveBackwards(driveTime):
  motor1.dc(-20)
  motor2.dc(-20)
  wait(driveTime)



def driveLeft(driveTime):
 motor1.dc(70) 
 motor2.dc(50) 
 wait(driveTime)

def driveRight(driveTime):
 motor1.dc(50) 
 motor2.dc(70) 
 wait(driveTime)
