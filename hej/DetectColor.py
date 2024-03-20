#!/usr/bin/env pybricks-micropython

#https://pybricks.com/ev3-micropython/ev3devices.html#gyroscopic-sensor
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import Motor, ColorSensor
from pybricks.parameters import Port, Button, Stop, Color
from pybricks.tools import wait
from pybricks.media.ev3dev import Font
from pybricks.media.ev3dev import ImageFile, Image


import time

from pybricks.tools import print, wait, StopWatch


def findWhite(ev3):
    sensor = ColorSensor(Port.S1)

    while True:
        color = sensor.color()
        print(color)  # Just to see the output
        if color == Color.WHITE:
            ev3.speaker.play_file('kung_fu.wav')
           # break  # Exits the loop if the color is white
        wait(10)  # Pause for a moment before the next reading

      


