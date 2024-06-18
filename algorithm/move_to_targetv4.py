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




class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0
        self.integral = 0

    def calculate(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.previous_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

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

def move_to_targetv4(camera_handler, ball):
    #ball.x, ball.y, ball.obstacle, ball.waypoints[x_størrelse].x, ball.waypoints[x_størrelse].y
    while(len(ball.waypoints)!=0):   
        print(ball)
        waypoint = ball.pop_waypoint()
        target_x, target_y = waypoint.x, waypoint.y
        position_threshold = 1   
        flag_done = 0
        while not(flag_done):
            image = camera_handler._run_video()
            image = transform(image)
            car = find_carv2(image)
            car = Car(car[0], car[1], car[2])
            
            # Initialize PID controllers
            Kp_angle, Ki_angle, Kd_angle = 0.001, 0, 0.05
            Kp_dist, Ki_dist, Kd_dist = 0.01, 0, 0.05
            angle_pid = PIDController(Kp_angle, Ki_angle, Kd_angle)
            dist_pid = PIDController(Kp_dist, Ki_dist, Kd_dist)
            
            # Commands
            comswallow = (0, 0, 0, 1, 0)

            # De-structure the target position
            #target_x, target_y = target_position
            
            #Hvis vi kører for langt sæt den op kører for kort sæt den ned
            
            #Virker nu!!!!!!!!
            angle_threshold = 4
            
            # Load car values into the car object
            current_x, current_y, current_angle = car.x, car.y, car.angle
            
            print(f"Desired position: {target_x,target_y}\nMy position: ({current_x}, {current_y}), angle: {current_angle}")
            
            # Calculate distance and desired angle
            distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
            
            desired_angle_rad = math.atan2(target_y-current_y, target_x-current_x)
            print(f"Desired angle in rad: {desired_angle_rad}")
            desired_angle =  math.degrees(desired_angle_rad)
            desired_angle = desired_angle % 360


            print(f"Desired angle in degrees:{desired_angle}\n")
            

            
            angle_error = (desired_angle - current_angle) % 360
            if angle_error > 180:
                angle_error -= 360
            
            if (distance < position_threshold) and (abs(angle_error)<angle_threshold):
                print("Target reached!")
                publish_controller_data(comswallow)  # Activate intake at target
                flag_done = 1
                break
                


            # Angle correction
            print(f"angle error: {abs(angle_error)}\n")
            if abs(angle_error) > angle_threshold:
                angle_correction = angle_pid.calculate(0, angle_error)
                if angle_error > 0:#Der forventes at skulle skrues her: 
                    publish_controller_data((0, 0, max(0.12, min(angle_correction, 1)), 0, 0))  # Tilt right
                else:
                    publish_controller_data((0, 0, max(-0.12, min(angle_correction, -1)), 0, 0))  # Tilt left
                continue
            
            # Forward movement control
            forward_speed = dist_pid.calculate(0, distance)
            forward_speed = max(0.15, min(forward_speed, 1))  # Clamp forward speed between 0.15 and 1
            publish_controller_data((0, forward_speed, 0, 0, 0))  # Move forward
            continue
        
  
    target_x, target_y = ball.x, ball.y
    position_threshold = 176 
    flag_done = 0
    while not(flag_done):
            image = camera_handler._run_video()
            image = transform(image)
            car = find_carv2(image)
            
            # Initialize PID controllers
            Kp_angle, Ki_angle, Kd_angle = 0.001, 0, 0.05
            Kp_dist, Ki_dist, Kd_dist = 0.01, 0, 0.05
            angle_pid = PIDController(Kp_angle, Ki_angle, Kd_angle)
            dist_pid = PIDController(Kp_dist, Ki_dist, Kd_dist)
            
            # Commands
            comswallow = (0, 0, 0, 1, 0)

            # De-structure the target position
            #target_x, target_y = target_position
            
            #Hvis vi kører for langt sæt den op kører for kort sæt den ned
            
            #Virker nu!!!!!!!!
            angle_threshold = 4
            
            # Load car values into the car object
            current_x, current_y, current_angle = car.x, car.y, car.angle
            
            print(f"Desired position: {target_x,target_y}\nMy position: ({current_x}, {current_y}), angle: {current_angle}")
            
            # Calculate distance and desired angle
            distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
            
            desired_angle_rad = math.atan2(target_y-current_y, target_x-current_x)
            print(f"Desired angle in rad: {desired_angle_rad}")
            desired_angle =  math.degrees(desired_angle_rad)
            desired_angle = desired_angle % 360


            print(f"Desired angle in degrees:{desired_angle}\n")
            

            
            angle_error = (desired_angle - current_angle) % 360
            if angle_error > 180:
                angle_error -= 360
            
            if (distance < position_threshold) and (abs(angle_error)<angle_threshold):
                print("Target reached!")
                publish_controller_data(comswallow)  # Activate intake at target
                flag_done = 1
                break
                


            # Angle correction
            print(f"angle error: {abs(angle_error)}\n")
            if abs(angle_error) > angle_threshold:
                angle_correction = angle_pid.calculate(0, angle_error)
                if angle_error > 0:#Der forventes at skulle skrues her: 
                    publish_controller_data((0, 0, max(0.12, min(angle_correction, 1)), 0, 0))  # Tilt right
                else:
                    publish_controller_data((0, 0, max(-0.12, min(angle_correction, -1)), 0, 0))  # Tilt left
                continue
            
            # Forward movement control
            forward_speed = dist_pid.calculate(0, distance)
            forward_speed = max(0.15, min(forward_speed, 1))  # Clamp forward speed between 0.15 and 1
            publish_controller_data((0, forward_speed, 0, 0, 0))  # Move forward
            continue
    
    
        
    

