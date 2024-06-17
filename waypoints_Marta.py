import math
import numpy as np
import matplotlib.pyplot as plt

class Waypoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Waypoint(x={self.x}, y={self.y})"

class Ball:
    def __init__(self, x, y, obstacle):
        self.x = x
        self.y = y
        self.obstacle = obstacle
        self.waypoints = []

    def add_waypoint(self, waypoint):
        print(f"Waypoint added at: {waypoint}, on {self}")
        self.waypoints.append(waypoint)

    def clear_waypoints(self):
        self.waypoints = []

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
    # def __init__(self,center, angle, x1, y1, x2, y2, x3, y3, x4, y4): # x1, y1, x2, y2 first line begins and ends.  x3, y3, x4, y4 where the second line begins and ends
    #     self.center = center
    #     self.angle = angle
    #     self.begin1 = (x1,y1)
    #     self.end1 = (x2,y2)
    #     self.begin2 = (x3,y3)
    #     self.end2 = (x4,y4)
   
    def __init__(self, center, angle, begin1, end1, begin2, end2):
        # Center and angle
        self.center = center
        self.angle = angle
        
        # Lines as tuples (x, y)
        self.begin1 = begin1
        self.end1 = end1
        self.begin2 = begin2
        self.end2 = end2
    
    
    def __repr__(self):
        return (f"Cross(center={self.center}, angle={self.angle}, "
                f"begin1={self.begin1}, end1={self.end1},begin2={self.begin2} end2={self.end2},")


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

def is_crossed_by_line(car, waypoint, cross_segments):
    for segment in cross_segments:
        if do_intersect((car.x, car.y), (waypoint.x, waypoint.y), segment[0], segment[1]):
            return True
    return False

def calc_cross_center(segments, rotation_angle_degrees):
    x_coords = [segment[0][0] for segment in segments] + [segment[1][0] for segment in segments]
    y_coords = [segment[0][1] for segment in segments] + [segment[1][1] for segment in segments]

    # Define the rotation matrix for the given rotation angle
    theta = np.radians(rotation_angle_degrees)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    return [np.mean(x_coords), np.mean(y_coords)], rotation_matrix

def rotate_point(point, center, rotation_matrix):
    translated_point = np.array(point) - np.array(center)
    rotated_point = np.dot(rotation_matrix, translated_point)
    return rotated_point + np.array(center)

def rotate_cross_segments(cross_segments, center, rotation_matrix):
    rotated_segments = []
    for segment in cross_segments:
        rotated_segment = [
            rotate_point(segment[0], center, rotation_matrix),
            rotate_point(segment[1], center, rotation_matrix)
        ]
        rotated_segments.append(rotated_segment)
    return rotated_segments

def get_quadrant(point, center):
    if point.x >= center[0] and point.y <= center[1]:
        return 1  # Top-right
    elif point.x <= center[0] and point.y <= center[1]:
        return 2  # Top-left
    elif point.x <= center[0] and point.y >= center[1]:
        return 3  # Bottom-left
    elif point.x >= center[0] and point.y >= center[1]:
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

def calc_obstacle_waypoints(ball, car, cross_segments):
    waypoint_distance = 150  # Distance from ball location to put waypoint
    cross_center, rotation_matrix = calc_cross_center(cross_segments, 30)
    rotated_cross_segments = rotate_cross_segments(cross_segments, cross_center, rotation_matrix)

    # if ball.y <= 100 and ball.x <= 100:
    #     angle = 45
    # else:
    #     if ball.y <= 100 and ball.x >= 1100:
    #      angle = 135
    if ball.y <= 100 and ball.x <= 100:
        angle = 45
    else:
        if ball.y <= 100 and ball.x >= 1100:
         angle = 135


    

    if ball.obstacle == 0:
        print("No waypoints added")
    else:
        if ball.obstacle == 1:
            angle = 45
        elif ball.obstacle == 2:
            angle = 135
    #     elif ball.obstacle == 3:
    #         angle = 315
    #     elif ball.obstacle == 4:
    #         angle = 225
    #     elif ball.obstacle == 5:
    #         angle = 90
    #     elif ball.obstacle == 6:
    #         angle = 0
    #     elif ball.obstacle == 7:
    #         angle = 270
    #     elif ball.obstacle == 8:
    #         angle = 180
    #     elif ball.obstacle == 9:
    #         angle = 0
    #     elif ball.obstacle == 10:
    #         angle = 270
    #     elif ball.obstacle == 11:
    #         angle = 180
    #     elif ball.obstacle == 12:
    #         angle = 90
    #     elif ball.obstacle == 13:
    #         angle = 225
    #     elif ball.obstacle == 14:
    #         angle = 135
    #     elif ball.obstacle == 15:
    #         angle = 45
    #     elif ball.obstacle == 16:
    #         angle = 315

        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)

        # Check if the cross is between the car and waypoint, if so, add additional waypoints
        while is_crossed_by_line(car, ball.waypoints[-1], rotated_cross_segments):
            add_additional_waypoint(ball, car, rotated_cross_segments, waypoint_distance, cross_center)

def add_additional_waypoint(ball, car, cross_segments, waypoint_distance, cross_center):
    first_waypoint = ball.waypoints[-1]
    first_quadrant = get_quadrant(first_waypoint, cross_center)
    closest_quadrants = get_closest_quadrants(first_quadrant)

    potential_quadrants = []
    for quadrant in closest_quadrants:
        if quadrant == 1:
            potential_quadrants.append((cross_center[0] + waypoint_distance, cross_center[1] - waypoint_distance))
        elif quadrant == 2:
            potential_quadrants.append((cross_center[0] - waypoint_distance, cross_center[1] - waypoint_distance))
        elif quadrant == 3:
            potential_quadrants.append((cross_center[0] - waypoint_distance, cross_center[1] + waypoint_distance))
        elif quadrant == 4:
            potential_quadrants.append((cross_center[0] + waypoint_distance, cross_center[1] + waypoint_distance))

    for x, y in potential_quadrants:
        waypoint = Waypoint(x, y)
        if not is_crossed_by_line(car, waypoint, cross_segments):
            ball.add_waypoint(waypoint)
            break

def plot_coordinates(car, ball, cross_segments):
    plt.figure(figsize=(10, 8))
    
    # Plot the car
    plt.plot(car.x, car.y, 'bo', label='Car')
    plt.text(car.x, car.y, 'Car', fontsize=12, ha='right')

    # Plot the ball
    plt.plot(ball.x, ball.y, 'ro', label='Ball')
    plt.text(ball.x, ball.y, 'Ball', fontsize=12, ha='right')

    cross_center, rotation_matrix = calc_cross_center(cross_segments, 30)
    rotated_segments = rotate_cross_segments(cross_segments, cross_center, rotation_matrix)

    crossbegin = []
    crossend = []
    #center, angle, x1, y1, x2, y2, x3, y3, x4, y4
    # Plot the rotated cross
    for segment in rotated_segments:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'r-')
        first = segment[0][0], segment[0][1]
        sec = segment[1][0], segment[1][1]
        print(f"Line begin (x,y) = {first}, line end (x,y) = {sec}")
        crossbegin.append(first)
        crossend.append(sec)
        plt.plot(segment[0][0], segment[0][1], 'rx')
        plt.plot(segment[1][0], segment[1][1], 'rx')
    plt.plot(cross_center[0], cross_center[1], 'rx', label='Rotated Cross Center')
    print(f"cross center {cross_center}")

    
    cross = Cross(cross_center,30, crossbegin[-2], crossend[-2], crossbegin[-1],  crossend[-1]) 
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

    return cross

def run():
    # Different balls
    # ball = Ball(33, 20, 1)  # Zone 1
    ball = Ball(1150, 50, 2)  # Zone 2
    #ball = Ball(29, 875, 3)  # Zone 3
    # ball = Ball(1150, 869, 4)  # Zone 4
    #ball = Ball(600, 30, 5)  # Zone 5
    #ball = Ball(33, 450, 6)  # Zone 6
    #ball = Ball(600, 862, 7)  # Zone 7
    #ball = Ball(1150, 450, 8)  # Zone 8
    #ball = Ball(598, 364, 9)  # Zone 9
    #ball = Ball(515, 449, 10)  # Zone 10
    #ball = Ball(598, 529, 11)  # Zone 11
    #ball = Ball(682, 449, 12)  # Zone 12
    # ball = Ball(580, 427, 13)  # Zone 13
    #ball = Ball(585, 466, 14)  # Zone 14
    #ball = Ball(612, 462, 15)  # Zone 15
    # ball = Ball(614, 427, 16)  # Zone 16
    # ball = Ball(639, 443, 16)     
    # ball = Ball(592, 396, 16)   
    # Car placements each corner
    #car = Car(200, 100, 0)  # Top left
    car = Car(200, 800, 0)  # Bottom left
    #car = Car(1000, 800, 0)  # Bottom right
    #car = Car(1000, 100, 0)  # Top right

    # Load cross segments from JSON file
    no_go_zones = [
        ((600, 380), (600, 520)),  # Arm 1 goes down
        ((530, 450), (670, 450))   # Arm 2 goes across
    ]
    
    calc_obstacle_waypoints(ball, car, no_go_zones)
    print(ball)

    # Plot the coordinates
    cross = plot_coordinates(car, ball, no_go_zones)
    print(cross)

run()
