import json
import math

class Waypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Ball:
    def __init__(self, x, y, obstacle):
        self.x = x
        self.y = y
        self.obstacle = obstacle
        self.waypoints = []

    def add_waypoint(self, waypoint):
        self.waypoints.append(waypoint)

    def clear_waypoints(self):
        self.waypoints = []
        
    def pop_waypoint(self):
        self.waypoints.pop

    def __repr__(self):
        return f"Ball(x={self.x}, y={self.y}, obstacle={self.obstacle}, waypoints={self.waypoints})"

def calc_obstacle_waypoints(ball):
    with open('no_go_zones.json', 'r') as file:
        data = json.load(file)
    cross_angle = data[2][1]
    cross_center = data[2][0]
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
        angle_rad = math.radians(cross_angle + 270)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad) + cross_length
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad) + cross_length
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 10:
        angle_rad = math.radians(cross_angle + 180)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad) + cross_length
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad) + cross_length
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 11:
        angle_rad = math.radians(cross_angle + 90)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad) + cross_length
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad) + cross_length
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 12:
        angle_rad = math.radians(cross_angle)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad) + cross_length
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad) + cross_length
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 13:
        angle_rad = math.radians(cross_angle + 45)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 14:
        angle_rad = math.radians(cross_angle + 45 + 90)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 15:
        angle_rad = math.radians(cross_angle + 45 + 90 * 2)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 16:
        angle_rad = math.radians(cross_angle + 45 + 90 * 3)
        waypoint_x = cross_center[0] + waypoint_distance * math.cos(angle_rad)
        waypoint_y = cross_center[1] + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)

    # Return the ball object (optional, since modifications are in-place)
    return ball

def other_waypoints():
    """
    if red cross between car and ball.waypoints
        # calculate waypoint quadrant
        if ball.waypoints.y < cross_center.y
            if ball.waypoints.x < cross_center.x 
                waypoint_quadrant = 0 # top_left
            elif ball.waypoints.x >= cross_center.x 
                waypoint_quadrant = 1 # top_right
        elif ball.waypoints.y >= cross_center.y
            if ball.waypoints.x < cross_center.x 
                waypoint_quadrant = 2 # bottom_left
            elif ball.waypoints.x >= cross_center.x 
                waypoint_quadrant = 3 # bottom_right

        # calculate car quadrant
        if car.y < cross_center.y
            if car.x < cross_center.x 
                car_quadrant = 0 # top_left
            elif car.x >= cross_center.x 
                car_quadrant = 1 # top_right
        elif car.y >= cross_center.y
            if car.x < cross_center.x 
                car_quadrant = 2 # bottom_left
            elif car.x >= cross_center.x 
                car_quadrant = 3 # bottom_right
            
        x,y = 1229, 900
        
        if car_quadrant % 2 == waypoint_quadrant % 2 # if they are both odd or even, they must be in opposite corners wich is worst case


    
    """
    

