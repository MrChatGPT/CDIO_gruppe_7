import json
import math
import matplotlib.pyplot as plt

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
    elif ball.obstacle == 13:
        angle = cross.angle + 45
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 14:
        angle = cross.angle + 45 + 90
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 15:
        angle = cross.angle + 45 + 90 * 2
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)
    elif ball.obstacle == 16:
        angle = cross.angle + 45 + 90 * 3
        if angle > 359:
            angle = angle -360
        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
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

    # Return the ball object (optional, since modifications are in-place)
    return ball

def plot_coordinates(car, ball, cross):
    plt.figure(figsize=(10, 8))
    
    # Plot the car
    plt.plot(car.x, car.y, 'bo', label='Car')
    plt.text(car.x, car.y, 'Car', fontsize=12, ha='right')

    # Plot the ball
    plt.plot(ball.x, ball.y, 'ro', label='Ball')
    plt.text(ball.x, ball.y, 'Ball', fontsize=12, ha='right')

    # Plot the cross center and arms
    for segment in cross.arms:
        plt.plot([segment.start[0], segment.end[0]], [segment.start[1], segment.end[1]], 'k-')
        plt.plot(segment.start[0], segment.start[1], 'kx')
        plt.plot(segment.end[0], segment.end[1], 'kx')
    plt.plot(cross.x, cross.y, 'kx', label='Cross Center')
    
    # Plot waypoints
    for i, waypoint in enumerate(ball.waypoints):
        plt.plot(waypoint.x, waypoint.y, 'go')
        plt.text(waypoint.x, waypoint.y, f'WP{i}', fontsize=12, ha='right')

    plt.xlim(0, 1229)
    plt.ylim(0, 900)
    plt.gca().invert_yaxis()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid()
    plt.show()

def get_waypoints(ball, car, cross):
    calc_obstacle_waypoints(ball, car, cross)
    plot_coordinates(car, ball, cross)

ball = Ball(600, 330, 16) 
car = Car(200, 800, 0)  # Bottom left

with open('no_go_zones.json', 'r') as file:
    data = json.load(file)
cross_angle = data[2][1]
cross_center = data[2][0]
arms = []
arms.append(data[0])
arms.append(data[1])
cross = Cross(cross_center[0], cross_center[1], cross_angle, arms)

get_waypoints(ball, car, cross)