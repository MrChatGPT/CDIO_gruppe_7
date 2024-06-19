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

def get_car_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    if data and isinstance(data, list) and len(data) > 0:
        car_data = data[0]
        if len(car_data) == 3:
            x, y, angle = car_data
            return Car(x, y, angle)
        else:
            raise ValueError("Invalid car data structure in JSON file.")
    else:
        raise ValueError("Invalid JSON structure.")

pid_forwards = PID(0.5, 0.0001, 0.001, setpoint=0)  # Starting with small PID coefficients
pid_forwards.output_limits = (-870, 870)

pid_turn = PID(0.2, 0.01, 0.001, setpoint=0)  # Starting with small PID coefficients
pid_turn.output_limits = (-158, 158) #180*0.88

def move_to_targetv5(camera_handler, ball):
    while len(ball.waypoints) != 0:   
        print("Før pop: ", ball)
        waypoint = ball.pop_waypoint()
        print("Efter pop: ", ball)
        target_x, target_y = waypoint.x, waypoint.y
        position_threshold = 10   
        flag_done = 0
        while not flag_done:
            image = camera_handler._run_video()
            image = transform(image)
            car = find_carv2(image)
            car = Car(car[0], car[1], car[2])
            
            # Initialize PID controllers
            angle_pid = PID(Kp=1, Ki=0, Kd=0, setpoint=0)
            dist_pid = PID(Kp=0, Ki=0, Kd=0, setpoint=0)
            
            # Commands
            comswallow = (0, 0, 0, 1, 0)
            
            angle_threshold = 1
            
            current_x, current_y, current_angle = car.x, car.y, car.angle
            
            distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
            
            desired_angle_rad = math.atan2(target_y - current_y, target_x - current_x)
            desired_angle = math.degrees(desired_angle_rad) % 360
            
            angle_error = (desired_angle - current_angle) % 360
            if angle_error > 180:
                angle_error -= 360
            
            if distance < position_threshold:
                flag_done = 1
                break
                
            if abs(angle_error) > angle_threshold:
                if abs(angle_error) > 100:
                    angle_correction = 0.4
                elif abs(angle_error) > 50:
                    angle_correction = 0.25
                else:
                    angle_correction = 0.11
                if angle_error > 0:
                    publish_controller_data((0, 0, angle_correction, 0, 0))  # Tilt right
                else:
                    publish_controller_data((0, 0, (-1 * angle_correction), 0, 0))  # Tilt left
                continue
            
            if distance > 800:
                forward_speed = 0.5
            elif distance > 500:
                forward_speed = 0.3
            else:
                forward_speed = 0.15
            publish_controller_data((0, forward_speed, 0, 0, 0))
            
    target_x, target_y = ball.x, ball.y
    position_threshold = 160
    flag_done = 0
    print("all waypoints cleared, next stop THE BALL!")
    while not flag_done:
        image = camera_handler._run_video()
        image = transform(image)
        car = find_carv2(image)
        car = Car(car[0], car[1], car[2])
        
        angle_pid = PID(Kp=0.1, Ki=0, Kd=0, setpoint=0)
        dist_pid = PID(Kp=0.001, Ki=0, Kd=0, setpoint=0)
        
        comswallow = (0, 0, 0, 1, 0)
        
        angle_threshold = 2
        
        current_x, current_y, current_angle = car.x, car.y, car.angle
        
        distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
        
        desired_angle_rad = math.atan2(target_y - current_y, target_x - current_x)
        desired_angle = math.degrees(desired_angle_rad) % 360
        
        angle_error = (desired_angle - current_angle) % 360
        if angle_error > 180:
            angle_error -= 360
        
        if distance < position_threshold and abs(angle_error) < angle_threshold:
            publish_controller_data(comswallow)
            flag_done = 1
            break
            
        if abs(angle_error) > angle_threshold:
            pid_output = -pid_turn(angle_error) / 180
            if pid_output < 0:
                pid_output = (pid_output - 0.12)
            elif pid_output > 0:
                pid_output = (pid_output + 0.12)
            publish_controller_data((0, 0, pid_output, 0, 0))
            continue
        pid_output = abs(pid_forwards(distance)/1000)
        pid_output = pid_output + 0.12
        
        publish_controller_data((0, pid_output, 0, 0, 0))
