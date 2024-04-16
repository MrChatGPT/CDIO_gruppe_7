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
import Motors
import socket

ev3 = EV3Brick()

# Set up UDP socket to listen on all available interfaces
udp_ip = ""  # Listen on all available interfaces
udp_port = 8888 # Portnummer
driveTime = 2000
drivePower = 70
pickupPower = 60

# Set up EV3 motors and drive time
Motors.InitMotors()
Motors.activateSmallMotors(pickupPower)

# Function to handle different actions based on the received letter
def handle_action(letter):
    if letter == b'U':
        Motors.driveStraight(driveTime, drivePower)
    elif letter == b'D':
        Motors.driveBackwards(driveTime, drivePower)
    elif letter == b'L':
        Motors.driveLeft(driveTime, drivePower)
    elif letter == b'R':
        Motors.driveRight(driveTime, drivePower)
    elif letter == b'S':
        Motors.deactivateLargeMotors()
    #New letters to be handled:
    elif letter == b'G':
        drivePower+=2
    elif letter == b'F':
        drivePower-=2
    elif letter == b's':
        Motors.deactivateSmallMotors()
    elif letter == b'g':
        pickupPower+=2
        Motors.activateSmallMotors(pickupPower)
    elif letter == b'f':
        pickupPower-=2
        Motors.activateSmallMotors(pickupPower)
    else:
        print("Invalid action:", letter.decode())

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # UDP

# Set socket to reuse address and port
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# Bind the socket to the port on all interfaces
sock.bind((udp_ip, udp_port))

# Listen for incoming UDP packets
print("UDP server is running...")
while True:
    data, addr = sock.recvfrom(1024)  # Receive data and sender's address
    print("Received letter:", data.decode())
    handle_action(data)
