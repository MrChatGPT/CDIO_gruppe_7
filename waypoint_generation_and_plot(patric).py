import math
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

def calc_cross_center(segments):
    x_coords = [point[0] for segment in segments for point in segment]
    y_coords = [point[1] for segment in segments for point in segment]
    center_x = sum(x_coords) / len(x_coords)
    center_y = sum(y_coords) / len(y_coords)
    return (center_x, center_y)

def get_quadrant(point, center):
    if point.x >= center[0] and point.y <= center[1]:
        return 1  # Top-right
    elif point.x <= center[0] and point.y <= center[1]:
        return 2  # Top-left
    elif point.x <= center[0] and point.y >= center[1]:
        return 3  # Bottom-left
    elif point.x >= center[0] and point.y >= center[1]:
        return 4  # Bottom-right

def calc_obstacle_waypoints(ball, car, cross_segments):
    cross_length = 100
    waypoint_distance = 150  # Distance from ball location to put waypoint
    cross_center = calc_cross_center(cross_segments)

    if ball.obstacle == 0:
        print("No waypoints added")
    else:
        angle = 45  # Default angle for obstacles 1-4, 13-16
        if ball.obstacle == 5:
            angle = 90
        elif ball.obstacle == 6:
            angle = 0
        elif ball.obstacle == 7:
            angle = 270
        elif ball.obstacle == 8:
            angle = 180
        elif ball.obstacle == 9:
            angle = 270
        elif ball.obstacle == 10:
            angle = 180
        elif ball.obstacle == 11:
            angle = 90
        elif ball.obstacle == 12:
            angle = 0
        elif ball.obstacle == 13:
            angle = 225  # 180 + 45
        elif ball.obstacle == 14:
            angle = 135  # 90 + 45
        elif ball.obstacle == 15:
            angle = 45  # 270 + 45
        elif ball.obstacle == 16:
            angle = 315  # 270 + 45

        angle_rad = math.radians(angle)
        waypoint_x = ball.x + waypoint_distance * math.cos(angle_rad)
        waypoint_y = ball.y + waypoint_distance * math.sin(angle_rad)
        waypoint = Waypoint(waypoint_x, waypoint_y)
        ball.add_waypoint(waypoint)

        # Check if the cross is between the car and waypoint, if so, add additional waypoints
        while is_crossed_by_line(car, ball.waypoints[-1], cross_segments):
            add_additional_waypoint(ball, car, cross_segments, cross_length, waypoint_distance, cross_center)

def add_additional_waypoint(ball, car, cross_segments, cross_length, waypoint_distance, cross_center):
    first_waypoint = ball.waypoints[-1]
    quadrant = get_quadrant(first_waypoint, cross_center)

    if quadrant == 1:
        nearby_quadrants = [(cross_center[0] + cross_length + waypoint_distance, cross_center[1] + cross_length + waypoint_distance),
                            (cross_center[0] - cross_length - waypoint_distance, cross_center[1] + cross_length + waypoint_distance)]
    elif quadrant == 2:
        nearby_quadrants = [(cross_center[0] + cross_length + waypoint_distance, cross_center[1] + cross_length + waypoint_distance),
                            (cross_center[0] + cross_length + waypoint_distance, cross_center[1] - cross_length - waypoint_distance)]
    elif quadrant == 3:
        nearby_quadrants = [(cross_center[0] + cross_length + waypoint_distance, cross_center[1] - cross_length - waypoint_distance),
                            (cross_center[0] - cross_length - waypoint_distance, cross_center[1] - cross_length - waypoint_distance)]
    elif quadrant == 4:
        nearby_quadrants = [(cross_center[0] - cross_length - waypoint_distance, cross_center[1] + cross_length + waypoint_distance),
                            (cross_center[0] - cross_length - waypoint_distance, cross_center[1] - cross_length - waypoint_distance)]

    for x, y in nearby_quadrants:
        waypoint = Waypoint(x, y)
        ball.add_waypoint(waypoint)
        if not is_crossed_by_line(car, waypoint, cross_segments):
            break

def plot_coordinates(car, ball, cross_segments):
    plt.figure(figsize=(10, 8))
    
    # Plot the car
    plt.plot(car.x, car.y, 'bo', label='Car')
    plt.text(car.x, car.y, 'Car', fontsize=12, ha='right')

    # Plot the ball
    plt.plot(ball.x, ball.y, 'ro', label='Ball')
    plt.text(ball.x, ball.y, 'Ball', fontsize=12, ha='right')

    # Plot the cross center and arms
    for segment in cross_segments:
        plt.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]], 'k-')
        plt.plot(segment[0][0], segment[0][1], 'kx')
        plt.plot(segment[1][0], segment[1][1], 'kx')
    cross_center = calc_cross_center(cross_segments)
    plt.plot(cross_center[0], cross_center[1], 'kx', label='Cross Center')
    
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


def run():
    #different balls:
    #ball = Ball(33, 20, 1) #zone 1
    #ball = Ball(1150, 50, 2) #zone 2
    #ball = Ball(29, 875, 3) #zone 3
    #ball = Ball(1150, 869, 4) #zone 4 
    #ball = Ball(600, 30, 5) #zone 5
    #ball = Ball(33, 450, 6) #zone 6
    #ball = Ball(600, 862, 7) #zone 7
    #ball = Ball(1150, 450, 8) #zone 8
    #ball = Ball(598, 364, 9) #zone 9
    #ball = Ball(515, 449, 10) #zone 10
    #ball = Ball(598, 529, 11) #zone 11
    #ball = Ball(682, 449, 12) #zone 12
    #ball = Ball(580, 427, 13) #zone 13
    #ball = Ball(585, 466, 14) #zone 14
    #ball = Ball(612, 462, 15) #zone 15
    ball = Ball(614, 427, 16) #zone 16

    #Car placements each corner:
    #car = Car(200, 100, 0) #top left
    car = Car(200, 800, 0) #bottom left
    #car = Car(1000, 800, 0) #bottom right
    #car = Car(1000, 100, 0) #top right

    # Load cross segments from JSON file
    no_go_zones = [
        ((600, 380), (600, 520)),  # Arm 1 goes down
        ((530, 450), (670, 450))   # Arm 2 goes across
    ]
    
    calc_obstacle_waypoints(ball, car, no_go_zones)
    print(ball)

    # Plot the coordinates
    plot_coordinates(car, ball, no_go_zones)

run()
