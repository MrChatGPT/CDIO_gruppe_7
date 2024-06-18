import json
from time import sleep
import numpy as np 
import os
#from algorithm.control import *
import math
from typing import Tuple, Optional


def publish_controller_data(command: Optional[Tuple[float, float, float, int, int]] = None):
    print("hej")


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

class Waypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Waypoint(x={self.x}, y={self.y})"

class Ball:
    def __init__(self, x, y, obstacle=0):
        self.x = x
        self.y = y
        self.obstacle = obstacle
        self.waypoints = []

    def add_waypoint(self, waypoint):
        print(f"Waypoint added at: {waypoint}, on {self}")
        self.waypoints.append(waypoint)

    def clear_waypoints(self):
        self.waypoints = []
        
    def pop_waypoint(self):
        self.waypoints.pop

    def __repr__(self):
        return f"Ball(x={self.x}, y={self.y}, obstacle={self.obstacle}, waypoints={self.waypoints})"

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle

    def __repr__(self):
        return f"Car(x={self.x}, y={self.y}, angle={self.angle})"

class Cross:
    class Arm:
        def __init__(self, start, end):
            self.start = start
            self.end = end

    def __init__(self, x, y, angle, arms):
        self.x = x
        self.y = y
        self.angle = angle
        self.arms = [self.Arm(*arm) for arm in arms] # lets fucking go! pointers i python, that is a first



# Function to read obstacle coordinates from a json file
def LoadObstacles(filename="no_go_zones.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
        cross_angle = data[2][1]
        cross_center = data[2][0]
        arms = []
        arms.append(data[0])
        arms.append(data[1])
        cross = Cross(cross_center[0], cross_center[1], cross_angle, arms)
    return cross

# Function to calculate the distance 
def Distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to determine the obstacle-value of a ball based on its position
def write_obstacle_val(ball, cross):
    
    # Defining the obstacle values for corners and edges of the track 
    TrackCorners = {
        (0, 0): 1,      # Top left corner 
        (1250, 0): 2,   # Top right corner  
        (0, 900): 3,    # Bottom left corner 
        (1250, 900): 4  # Bottom right corner 
    } 

    TrackEdges = {
        ((0, 0), (1250, 0)): 5,      # Top edge
        ((0, 0), (0, 900)): 6,       # Left edge
        ((0, 900), (1250, 900)): 7,  # Bottom edge
        ((1250, 0), (1250, 900)): 8  # Right edge
    }

    # Converting the nested list that represents the points of the cross to a list
    
    print(cross.arms[1].end)
    print(cross.arms[1].start)
    print(cross.arms[0].end)
    print(cross.arms[0].start)
    # Defining the obstacle values inside the cross
    ObstaclePoints = {
        tuple(cross.arms[1].end): 9,  # Top point 
        tuple(cross.arms[1].start): 10,  # Bottom point  
        tuple(cross.arms[0].end): 11,  # Left point 
        tuple(cross.arms[0].start): 12   # Right point
    }

    # Check proximity to track corners
    for point, value in TrackCorners.items():
        if Distance((ball.x, ball.y), point) < 100:  # Arbitrary distance to be considered "close"
            ball.obstacle = value
            return
    
    # Check proximity to track edges
    for ((x1, y1), (x2, y2)), value in TrackEdges.items():
        if x1 == x2:  # Vertical edge
            if abs(ball.x - x1) < 100:  # Vertical edge proximity
                ball.obstacle = value
                return
        elif y1 == y2:  # Horizontal edge
            if abs(ball.y - y1) < 100:  # Horizontal edge proximity
                ball.obstacle = value
                return
            
    # Determine which inner quadrant the ball is in
    TopPoint = tuple(cross.arms[1].end)
    BottomPoint = tuple(cross.arms[1].start)
    LeftPoint = tuple(cross.arms[0].end)
    RightPoint = tuple(cross.arms[0].start)

    # Check if the ball is close to lines forming the inner quadrants
    if Distance((ball.x, ball.y), TopPoint) < 100 and Distance((ball.x, ball.y), (cross.x, cross.y)) < 100 and Distance((ball.x, ball.y), LeftPoint) < 100:
        ball.obstacle = 13
        return
    
    # Check if the ball is in the top-right inner corner
    elif Distance((ball.x, ball.y), TopPoint) < 100 and Distance((ball.x, ball.y), (cross.x, cross.y)) < 100 and Distance((ball.x, ball.y), RightPoint) < 100:
        ball.obstacle = 16
        return
    
    # Check if the ball is in the bottom-left inner corner
    elif Distance((ball.x, ball.y), LeftPoint) < 100 and Distance((ball.x, ball.y), (cross.x, cross.y)) < 100 and Distance((ball.x, ball.y), BottomPoint) < 100:
        ball.obstacle = 14
        return
    
    # Check if the ball is in the bottom-right inner corner
    elif Distance((ball.x, ball.y), BottomPoint) < 100 and Distance((ball.x, ball.y), (cross.x, cross.y)) < 100 and Distance((ball.x, ball.y), RightPoint) < 100:
        ball.obstacle = 15
        return
    
    # Check proximity to obstacle points (outer points of the cross)
    for point, value in ObstaclePoints.items():
        if Distance((ball.x, ball.y), point) < 100:  # Arbitrary distance to be considered "close"
            ball.obstacle = value
            return

    # Default value for balls not close to any obstacle (track or cross)
    return 0


def do_intersect(p1, q1, p2, q2):
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0
        return 1 if val > 0 else 2

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, p2, q1):
        return True

    if o2 == 0 and on_segment(p1, q2, q1):
        return True

    if o3 == 0 and on_segment(p2, p1, q2):
        return True

    if o4 == 0 and on_segment(p2, q1, q2):
        return True

    return False

def is_crossed_by_line(car, waypoint, cross):
    for segment in cross.arms:
        if do_intersect((car.x, car.y), (waypoint.x, waypoint.y), segment.start, segment.end):
            return True
    return False

def get_quadrant(point, x, y):
    if point.x >= x and point.y <= y:
        return 1  # Top-right
    elif point.x <= x and point.y <= y:
        return 2  # Top-left
    elif point.x <= x and point.y >= y:
        return 3  # Bottom-left
    elif point.x >= x and point.y >= y:
        return 4  # Bottom-right

def get_closest_quadrants(first_quadrant):
    if first_quadrant == 1:
        return [2, 4]
    elif first_quadrant == 2:
        return [1, 3]
    elif first_quadrant == 3:
        return [2, 4]
    elif first_quadrant == 4:
        return [1, 3]

def calc_obstacle_waypoints(ball, car, cross):
    cross_length = 100
    waypoint_distance = 150 # Distance from ball location to put waypoint
    x,y = 1229, 900

    if ball.obstacle == 0:
        print("No waypoints added")
    elif ball.obstacle == 1:
        waypoint = Waypoint(waypoint_distance, waypoint_distance)
        ball.add_waypoint(waypoint)
        # if cross between bot and waypoint, make another point
    elif ball.obstacle == 2:
        waypoint = Waypoint(x - waypoint_distance, waypoint_distance)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 3:
        waypoint = Waypoint(x - waypoint_distance, y - waypoint_distance)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 4:
        waypoint = Waypoint(waypoint_distance, y - waypoint_distance)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 5:
        waypoint = Waypoint(ball.x, ball.y + waypoint_distance)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 6:
        waypoint = Waypoint(ball.x + waypoint_distance, ball.y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 7:
        waypoint = Waypoint(ball.x, ball.y - waypoint_distance)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 8:
        waypoint = Waypoint(ball.x - waypoint_distance, ball.y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 9:
        angle = cross.angle + 270
        if angle > 359:
            angle = angle - 360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 10:
        angle = cross.angle + 180
        if angle > 359:
            angle = angle - 360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 11:
        angle = cross.angle + 90
        if angle > 359:
            angle = angle - 360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 12:
        angle = cross.angle
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 15:
        angle = cross.angle + 45
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = cross.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 14:
        angle = cross.angle + 45 + 90
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = cross.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 13:
        angle = cross.angle + 45 + 90 * 2
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = cross.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 16:
        angle = cross.angle + 45 + 90 * 3
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = cross.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    
    while is_crossed_by_line(car, ball.waypoints[-1], cross):
            add_additional_waypoint(ball, car, cross, waypoint_distance)

def add_additional_waypoint(ball, car, cross, waypoint_distance):
    first_waypoint = ball.waypoints[-1]
    first_quadrant = get_quadrant(first_waypoint, cross.x, cross.y)
    closest_quadrants = get_closest_quadrants(first_quadrant)

    potential_quadrants = []
    for quadrant in closest_quadrants:
        if quadrant == 1:
            potential_quadrants.append((cross.x + waypoint_distance, cross.y - waypoint_distance))
        elif quadrant == 2:
            potential_quadrants.append((cross.x - waypoint_distance, cross.y - waypoint_distance))
        elif quadrant == 3:
            potential_quadrants.append((cross.x - waypoint_distance, cross.y + waypoint_distance))
        elif quadrant == 4:
            potential_quadrants.append((cross.x + waypoint_distance, cross.y + waypoint_distance))

    for x, y in potential_quadrants:
        waypoint = Waypoint(x, y)
        if not is_crossed_by_line(car, waypoint, cross):
            ball.add_waypoint(waypoint)
            break

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

def move_to_targetv3(target_position):

    #Find and create a ball object for target_position
    ball = Ball(target_position[0],target_position[1])
    print(f"Ball before obstacle values: {ball}")
    car = get_car_data_from_json(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'robot.json'))
    print(car)
    #Set an obstacle number:
    cross = LoadObstacles()
    write_obstacle_val(ball, cross)
    print(f"Ball after obstacle values: {ball}")
    #calculate waypoints (even if obstacle == 0, if last ball is on opposite site of cross)
    calc_obstacle_waypoints(ball, car, cross)
    
    print(f"Ball after obstacle values, and waypoints: {ball}")
    if len(ball.waypoints) == 0:
        target_x, target_y = ball.x, ball.y

    

    # print(f"Waypoint x: {ball.waypoints[(len(ball.waypoints)-1)].x}\n")
    # print(f"Waypoint y:{ball.waypoints[(len(ball.waypoints)-1)].y}\n")


    if car.x == ball.waypoints[(len(ball.waypoints)-1)].x and car.y == ball.waypoints[(len(ball.waypoints)-1)].y:
        ball.waypoints.remove[(len(ball.waypoints)-1)]
    
    if car.x != ball.waypoints[(len(ball.waypoints)-1)].x or car.y != ball.waypoints[(len(ball.waypoints)-1)].y:
        target_x, target_y = ball.waypoints[(len(ball.waypoints)-1)].x, ball.waypoints[(len(ball.waypoints)-1)].y

    print(f"Target x,y = {target_x, target_y}")


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
    position_threshold = 176 
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
        return 1


    # Angle correction
    print(f"angle error: {abs(angle_error)}\n")
    if abs(angle_error) > angle_threshold:
        angle_correction = angle_pid.calculate(0, angle_error)
        if angle_error > 0:#Der forventes at skulle skrues her: 
            publish_controller_data((0, 0, max(0.12, min(angle_correction, 1)), 0, 0))  # Tilt right
        else:
            publish_controller_data((0, 0, max(-0.12, min(angle_correction, -1)), 0, 0))  # Tilt left
        return 0
    
    # Forward movement control
    forward_speed = dist_pid.calculate(0, distance)
    forward_speed = max(0.15, min(forward_speed, 1))  # Clamp forward speed between 0.15 and 1
    publish_controller_data((0, forward_speed, 0, 0, 0))  # Move forward
    return 0


move_to_targetv3((700,400))