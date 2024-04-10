#!/usr/bin/env pybricks-micropython
from pybricks.hubs import EV3Brick
from pybricks.ev3devices import (Motor, TouchSensor, ColorSensor,
                                 InfraredSensor, UltrasonicSensor, GyroSensor)
from pybricks.parameters import Port, Stop, Direction, Button, Color
from pybricks.tools import wait, StopWatch, DataLog
from pybricks.robotics import DriveBase
from pybricks.media.ev3dev import SoundFile, ImageFile
import socket

# This program requires LEGO EV3 MicroPython v2.0 or higher.
# Click "Open user guide" on the EV3 extension tab for more information.


# Create your objects here.
ev3 = EV3Brick()

# Write your program here.
ev3.speaker.beep()




# Initializing motors 
LeftMotor = Motor(Port.A)
RightMotor = Motor(Port.B)


# Setting server
ev3_ip = '0.0.0.0'  # Listen on all interfaces
ev3_port = 7777     # Port number

# Creating UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Binding the socket to the address
sock.bind((ev3_ip, ev3_port))


# For debugging purposes 
print("Listening on {} for UDP packets...".format(ev3_port))


try:
    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        message = data.decode()
        print("Received message:", message)

        if message == 'forward':
            LeftMotor.run(360) # Speed
            RightMotor.run(360)
            wait(1000) # Delay  
            LeftMotor.stop()
            RightMotor.stop()
        elif message == 'backward':
            LeftMotor.run(-360)
            RightMotor.run(-360)
            wait(1000)  
            LeftMotor.stop()
            RightMotor.stop()
        elif message == 'left':
            LeftMotor.run(-180)
            RightMotor.run(180)
            wait(1000)  
            LeftMotor.stop()
            RightMotor.stop()
        elif message == 'right':
            LeftMotor.run(180)
            RightMotor.run(-180)
            wait(1000)  
            LeftMotor.stop()
            RightMotor.stop()
        else:
            print("ERROR!! Unknown command!!") # For debugging purposes 
finally:
    print("Closing socket")
    sock.close()