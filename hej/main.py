#!/usr/bin/env pybricks-micropython

#https://pybricks.com/ev3-micropython/ev3devices.html#gyroscopic-sensor
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, ColorSensor
from pybricks.parameters import Port, Button, Stop
from pybricks.tools import wait
from pybricks.media.ev3dev import Font
from pybricks.media.ev3dev import ImageFile, Image
from pybricks.tools import print, wait, StopWatch


import time
import Motors, FunnyBunny, DetectColor


ev3 = EV3Brick()

#Welcome greeting
#ev3.screen.load_image(ImageFile.EVIL)

# ev3.screen.load_image(ImageFile.EVIL)
# hello = "go fuck yourself"
# ev3.screen.load_image("sponge2")
# hello = "The names is Bond. James Bond"
# ev3.speaker.say(hello)
# wait(2000)
# ev3.screen.clear()
    



#############################################################################
##Output C is for the right motor on the car, B is for the left motor.     ##
##A is for the medium/small motor                                          ##
##Uncomment all the places with motor0 when running the program on thursday##
#############################################################################



# FunnyBunny.Funny(ev3)
# Motors.InitMotors()
# driveTime = 15000
# Motors.driveStraight(driveTime)
# Motors.driveBackwards(driveTime)
# Motors.driveLeft(driveTime)
# Motors.driveRight(driveTime)

#Test of ColorSensor
DetectColor.findWhite(ev3)




pressed_buttons = ev3.buttons.pressed()



