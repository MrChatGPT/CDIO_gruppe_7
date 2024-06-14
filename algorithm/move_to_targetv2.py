import json
from time import sleep
import numpy as np 
import os
from algorithm.control import publish_controller_data
import math

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


class Car:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def __repr__(self):
        return f"Car(x={self.x}, y={self.y}, angle={self.angle})"

def move_to_targetv2(target_position):
    # Initialize PID controllers
    Kp_angle, Ki_angle, Kd_angle = 0.001, 0, 0.05
    Kp_dist, Ki_dist, Kd_dist = 0.01, 0, 0.05
    angle_pid = PIDController(Kp_angle, Ki_angle, Kd_angle)
    dist_pid = PIDController(Kp_dist, Ki_dist, Kd_dist)
    
    # Commands
    comstop = (0, 0, 0, 0, 0)
    #comtiltleft = (0, 0, -0.15, 0, 0)
    #comtiltright = (0, 0, 0.15, 0, 0)
    #comforward = (0, 0.15, 0, 0, 0)
    comswallow = (0, 0, 0, 1, 0)

    # Get the project's root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    json_file_path = os.path.join(project_root, 'robot.json')

    # De-structure the target position
    target_x, target_y = target_position
    position_threshold = 185 #gammel 190
    angle_threshold = 11
    
    while True:
        # Load car values into the car object
        car = get_car_data_from_json(json_file_path)
        current_x, current_y, current_angle = car.x, car.y, car.angle
        
        print(f"Desired position: {target_position}\nMy position: ({current_x}, {current_y}), angle: {current_angle}")
        
        # Calculate distance and desired angle
        distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
        
        desired_angle = (math.degrees(math.atan2(target_y - current_y, target_x - current_x))-90) % 360
        print(f"Desired angle:{desired_angle}\n")
        
        angle_error = (desired_angle - current_angle) % 360
        if angle_error > 180:
            angle_error -= 360
        
        if (distance < position_threshold) and (abs(angle_error)<angle_threshold):
            print("Target reached!")
            publish_controller_data(comswallow)  # Activate intake at target
            #sleep(0.2)
            #publish_controller_data(comstop)
            return 1


        # Angle correction
        print(f"angle error: {abs(angle_error)}\n")
        if abs(angle_error) > angle_threshold:
            angle_correction = angle_pid.calculate(0, angle_error)
            if angle_error > 180:
                publish_controller_data((0, 0, max(0.12, min(angle_correction, 1)), 0, 0))  # Tilt right
            else:
                publish_controller_data((0, 0, max(-0.12, min(angle_correction, -1)), 0, 0))  # Tilt left
            #sleep(0.2)  # Allow time for turning
            #publish_controller_data(comstop)
            return 0
        
        # Forward movement control
        forward_speed = dist_pid.calculate(0, distance)
        forward_speed = max(0.15, min(forward_speed, 1))  # Clamp forward speed between 0.15 and 1
        publish_controller_data((0, forward_speed, 0, 0, 0))  # Move forward
        #sleep(0.2)  # Allow time for moving
        #publish_controller_data(comstop)
        return 0

# Example usage
# target_position = (target_x, target_y)
# move_to_target(target_position)
