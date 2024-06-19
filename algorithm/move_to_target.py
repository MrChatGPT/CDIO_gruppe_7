import json
from time import sleep
import numpy as np
import os
from algorithm.control import *
import math
from typing import Tuple, Optional
from picture.livefeed import *
from picture.transform_arena import *
from picture.image_detection import *
from simple_pid import PID

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle

    def __repr__(self):
        return f"Car(x={self.x}, y={self.y}, angle={self.angle})"
    
class Waypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Waypoint(x={self.x}, y={self.y})"

pid_forwards = PID(0.5, 0.0001, 0.001, setpoint=0)  # Starting with small PID coefficients
pid_forwards.output_limits = (-870, 870)

pid_turn = PID(0.2, 0.01, 0.001, setpoint=0)  # Starting with small PID coefficients
pid_turn.output_limits = (-158, 158) #180*0.88

def move(camera_handler, waypoint, position_threshold, angle_threshold):
    while True:
        image = camera_handler._run_video()
        image = transform(image)
        car = find_carv2(image)
        car = Car(car[0], car[1], car[2])

        distance = math.sqrt((waypoint.x - car.x) ** 2 + (waypoint.y - car.y) ** 2)

        desired_angle_rad = math.atan2(waypoint.y - car.y, waypoint.x - car.x)
        desired_angle = math.degrees(desired_angle_rad) % 360
        angle_error = (desired_angle - car.angle) % 360
        if angle_error > 180:
            angle_error -= 360
        
        if distance < position_threshold and abs(angle_error) < angle_threshold:
            publish_controller_data((0, 0, 0, 1, 0))
            return
                    
        if abs(angle_error) > angle_threshold:
            pid_output = - pid_turn(angle_error) / 250 # now it should go between -0.72 and +0.72

            if pid_output < 0:
                pid_output = (pid_output - 0.12) # at most -0.84

            elif pid_output > 0:
                pid_output = (pid_output + 0.12)

            if angle_error < 5: # ultra slow motion activated
                publish_controller_data((0, 0, pid_output, 0, 0))
                time.sleep(0.05)
                publish_controller_data((0, 0, pid_output, 0, 0))
                continue

            publish_controller_data((0, 0, pid_output, 0, 0))
            continue

        pid_output = abs(pid_forwards(distance - position_threshold)/1000)
        pid_output = pid_output + 0.12
        
        publish_controller_data((0, pid_output, 0, 0, 0))


def move_to_target(camera_handler, ball):
    # handling all waypoints
    while len(ball.waypoints) != 0:   
        waypoint = ball.pop_waypoint()
        print("Going for waypoint at:", waypoint)
        move(camera_handler, waypoint, position_threshold=200, angle_threshold=7)
    
    print("all waypoints cleared, next stop THE BALL!")
    waypoint = Waypoint(ball.x, ball.y)
    move(camera_handler, waypoint, position_threshold=150, angle_threshold=0.4)
    
    print("Ball has been collected")
