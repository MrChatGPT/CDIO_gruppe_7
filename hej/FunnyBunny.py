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




def Funny(ev3):
  ev3.screen.clear()
  ev3.screen.load_image(ImageFile.PINCHED_LEFT)
  wait(1000)
  ev3.screen.load_image(ImageFile.PINCHED_RIGHT)
  ev3.speaker.play_file('motor_start.wav')
  ev3.screen.load_image(ImageFile.PINCHED_MIDDLE)