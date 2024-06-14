import tkinter as tk
from math import radians, cos, sin
from random import randint
import time
import numpy as np
import math


class Car:

    def __init__(self, canvas, x, y, angle):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.angle = angle
        self.turn = 0
        self.shape = None
        self.wheels = []
        self.sensors = []
        self.tkimage = None
        self.canvas_obj = None
        self.rotation_direction = 0  # 1 for clockwise, -1 for counterclockwise
        self.is_rotating = False  # Flag to control rotation

    def rotate(self):
        self.angle = (self.angle + self.rotation_direction) % 360  # Change direction based on rotation_direction
        print(f"Current angle: {self.angle}")  # Print the current angle
        self.update_position()
        self.canvas.after(100, self.rotate)

    def update_position(self):
        rad_angle = radians(self.angle)
        cos_a = cos(rad_angle)
        sin_a = sin(rad_angle)

        # Calculate new coordinates for the car body
        half_length = 50
        half_width = 50
        corners = [
            (self.x - half_length, self.y - half_width),
            (self.x + half_length, self.y - half_width),
            (self.x + half_length, self.y + half_width),
            (self.x - half_length, self.y + half_width)
        ]
        rotated_corners = [(self.x + (x - self.x) * cos_a - (y - self.y) * sin_a,
                            self.y + (x - self.x) * sin_a + (y - self.y) * cos_a)
                           for x, y in corners]

        self.canvas.coords(self.shape,
                           *rotated_corners[0],
                           *rotated_corners[1],
                           *rotated_corners[2],
                           *rotated_corners[3])

        # Calculate new coordinates for the wheels
        wheel_positions = [
            (self.x - half_length , self.y - half_width),
            (self.x + half_length , self.y - half_width),
            (self.x - half_length , self.y + half_width),
            (self.x + half_length , self.y + half_width)
        ]
        rotated_wheels = [(self.x + (x - self.x) * cos_a - (y - self.y) * sin_a,
                           self.y + (x - self.x) * sin_a + (y - self.y) * cos_a)
                          for x, y in wheel_positions]
        for wheel, pos in zip(self.wheels, rotated_wheels):
            self.canvas.coords(wheel, pos[0] - 10, pos[1] - 10, pos[0] + 10, pos[1] + 10)

        # Calculate new coordinates for the sensors
        sensor_positions = [(self.x-10, self.y+40), (self.x +15  , self.y+ 40)]
        rotated_sensors = [(self.x + (x - self.x) * cos_a - (y - self.y) * sin_a,
                            self.y + (x - self.x) * sin_a + (y - self.y) * cos_a)
                           for x, y in sensor_positions]
        for sensor, pos in zip(self.sensors, rotated_sensors):
            self.canvas.coords(sensor, pos[0] - 5, pos[1] - 5, pos[0] + 5, pos[1] + 5)

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




def update_mouse_coordinates(event, label):
    label.config(text=f"Mouse Coordinates: x={event.x}, y={event.y}")

def draw_car(canvas, car):
    car.shape = canvas.create_polygon(car.x - 50, car.y - 50, car.x + 50, car.y - 50,
                                      car.x + 50, car.y + 50, car.x - 50, car.y + 50,
                                      outline="black", width=2, fill='darkgrey')
    car.wheels = [
        canvas.create_oval(car.x - 60, car.y - 60, car.x - 40, car.y - 40, outline="black", width=2, fill='black'),
        canvas.create_oval(car.x + 40, car.y - 60, car.x + 60, car.y - 40, outline="black", width=2, fill='black'),
        canvas.create_oval(car.x - 60, car.y + 40, car.x - 40, car.y + 60, outline="black", width=2, fill='black'),
        canvas.create_oval(car.x + 40, car.y + 40, car.x + 60, car.y + 60, outline="black", width=2, fill='black')
    ]
    car.sensors = [
        canvas.create_oval(car.x + 25, car.y + 40, car.x + 35, car.y + 50, outline="black", width=2, fill='green'),
        canvas.create_oval(car.x + 70, car.y + 40, car.x + 80, car.y + 50, outline="black", width=2, fill='green')
    ]

# def move_to_target(car, target_position, green_dot_y_range, canvas):
#     current_x, current_y = car.x, car.y
#     target_x, target_y = target_position
#     target_x += 10
#     target_y -= 50

#     # comstop = (0, 0)
#     # comtiltleft = (0, -5)
#     # comtiltright = (0, 5)
#     # comforward = (5, 0)

#     threshhold = 20  # Adjusted threshold

#     # Calculate the differences
#     dx = target_x - current_x
#     dy = target_y - current_y

#     print(f"current x,y {current_x},{current_y}")

#     # Calculate the angle to the target
#     desired_angle = math.atan2(dy, dx) * (180 / math.pi)  # Convert radians to degrees
#     if desired_angle < 0:
#         desired_angle += 360

#     current_angle = car.angle  # Assuming car.angle is in degrees

#     print(f"Current position: ({current_x}, {current_y}), Target position: ({target_x}, {target_y})")
#     print(f"Current angle: {current_angle}, Desired angle: {desired_angle}")

#     angle_diff = (desired_angle - current_angle + 360) % 360
#     if angle_diff > 180:
#         angle_diff -= 360

#     # Determine the movement based on the angle difference
#     if abs(dx) <= threshhold and abs(dy) <= threshhold and abs(angle_diff) < 10:
#         car.is_rotating = False
#         print(f"Stopping, within threshold and aligned. dx: {dx}, dy: {dy}, angle_diff: {angle_diff}")
#         car.angle = 0
#         return comstop  # The car is close to the target position and aligned
#     else:
#         if abs(angle_diff) < 20:  # If the angle difference is small, move forward
#             car.is_rotating = False
#             print(f"Moving forward, angle diff is {angle_diff}")
#             return comforward
#         else:
#             car.is_rotating = True
#             if angle_diff > 0:
#                 print(f"Rotating clockwise, angle diff is {angle_diff}")
#                 car.rotation_direction = 1
#                 return comtiltright  # Rotate clockwise
#             else:
#                 print(f"Rotating counter-clockwise, angle diff is {angle_diff}")
#                 car.rotation_direction = -1
#                 return comtiltleft  # Rotate anti-clockwise

def draw_rectangle(canvas):
    canvas.create_rectangle(10, 10, 1250, 900, outline="red", width=10)
    center_x = 1250 // 2
    center_y = 900 // 2
    cross_size = 40
    canvas.create_line(center_x - cross_size, center_y, center_x + cross_size, center_y, fill="red", width=5)
    canvas.create_line(center_x, center_y - cross_size, center_x, center_y + cross_size, fill="red", width=5)

def move_car(canvas, car, command):
    dx, dy = command
    rad_angle = radians(car.angle)
    dx_rot = dx * cos(rad_angle) - dy * sin(rad_angle)
    dy_rot = dx * sin(rad_angle) + dy * cos(rad_angle)

    new_x = car.x + dx_rot
    new_y = car.y + dy_rot

    # Define the rectangle boundaries
    left_bound = 10
    right_bound = 1250 - 10
    top_bound = 10
    bottom_bound = 900 - 10
    toplx, toply = 12,15
    toprx, topry = 1240, 15
    bottomlx, bottomly = 15 , 894
    bottomrx, bottomry = 1244 , 880


    # tic = time.perf_counter()


    dx, dy = command
    canvas.move(car.shape, dx, dy)
    for wheel in car.wheels:
        canvas.move(wheel, dx, dy)
    for sensor in car.sensors:
        canvas.move(sensor, dx, dy)
    car.x += dx
    car.y += dy

    # toc = time.perf_counter()
    # print(f"moved car in {toc - tic:0.4f} seconds")

# def animate_car(canvas, car, targetX, targetY, coord_label, green_dot_y_range):
#     command = move_to_target(car, (targetX, targetY), green_dot_y_range, canvas)
#     move_car(canvas, car, command)
#     coord_label.config(text=f"Car Coordinates: x={car.x}, y={car.y}, angle={car.angle}")
#     canvas.after(250, animate_car, canvas, car, targetX, targetY, coord_label, green_dot_y_range)

def animate_car(canvas, car, targetX, targetY, coord_label):
    if move_to_targetv2((targetX, targetY), car, canvas, 0, 0) == 0:
        coord_label.config(text=f"Car Coordinates: x={car.x}, y={car.y}, angle={car.angle}")
        canvas.after(100, animate_car, canvas, car, targetX, targetY, coord_label)

def is_within_boundaries(new_x, new_y):
    left_bound = 30  #10
    right_bound = 1250 - 70 #30
    top_bound = 30   #10
    bottom_bound = 900 - 70 #30
    return left_bound <= new_x <= right_bound and top_bound <= new_y <= bottom_bound


def stop_rotation(car):
    car.is_rotating = False
    car.rotation_direction = 0

def key_handler(event, car, canvas):
   
    key = event.keysym
    comstop = (0, 0)
    comtiltleft = (-4, 0)
    comtiltright = (4, 0)
    # comforward = (0, 5)



    curAngle = car.angle 



    if key == 'a':

        car.is_rotating = True
        car.rotation_direction = 1
        canvas.after(2000, lambda: stop_rotation(car))  # Schedule stop after 200ms
        command = comstop


        
        
    elif key == 'd':
        car.is_rotating = True
        car.rotation_direction = -1
        # tic = time.perf_counter()
        canvas.after(2000, lambda: stop_rotation(car))  # Schedule stop after 200ms
        # toc = time.perf_counter()
        # print(f"total time {toc - tic:0.4f} seconds, car angle is {car.angle}")
        command = comstop
        # car.angle = (car.angle + 5) % 360
        # command = comtiltright
    elif key == 'w':
        if curAngle == 0:
            comforward = (0, 2)
        elif curAngle == 90:
            comforward = (-4, 0)
        elif curAngle == 180:
            comforward = (0, -4)
        elif curAngle == 270:
            comforward = (4, 0)
        
        elif 0 < curAngle < 30:
            comforward = (-2, 4)
        elif 30 <= curAngle < 60:
            comforward = (-2, 2)
        elif 60 <= curAngle < 90:
            comforward = (-4, 4)
        elif 90 < curAngle < 120:
            comforward = (-4, -2)
        elif 120 <= curAngle < 150:
            comforward = (-4, -4)
        elif 150 <= curAngle < 180:
            comforward = (-2, -4)
        elif 180 < curAngle < 210:
            comforward = (2, -4)
        elif 210 <= curAngle < 240:
            comforward = (2, -4)
        elif 240 <= curAngle < 270:
            comforward = (4, -4)
        elif 270 < curAngle < 300:
            comforward = (4, 2)
        elif 300 <= curAngle < 330:
            comforward = (4, 4)
        elif 330 <= curAngle <= 360:
            comforward = (2, 4)
     
       
        
        command = comforward
        print(f"comforward{comforward}")

    elif key == 's':
        command = comstop
    
    elif key == 'f':
        car.is_rotating = False
        car.rotation_direction = 0
        command = comstop  # Ensure command is defined, though it may not be used

    elif key == 't':
        car.is_rotating = True
        car.rotation_direction = 1
        command = comstop  # Ensure command is defined, though it may not be used

    else:
        return


    dx, dy = command
    rad_angle = radians(car.angle)
    dx_rot = dx * cos(rad_angle) - dy * sin(rad_angle)
    dy_rot = dx * sin(rad_angle) + dy * cos(rad_angle)

    new_x = car.x + dx_rot
    new_y = car.y + dy_rot

    if is_within_boundaries(new_x, new_y):
        move_car(canvas, car, command)

    # move_car(canvas, car, command)


# def publish_controller_data(tuple(float, float, float, int, int)):
#     dx, dy = command
#     rad_angle = radians(car.angle)
#     dx_rot = dx * cos(rad_angle) - dy * sin(rad_angle)
#     dy_rot = dx * sin(rad_angle) + dy * cos(rad_angle)

#     new_x = car.x + dx_rot
#     new_y = car.y + dy_rot

#     # Define the rectangle boundaries
#     left_bound = 10
#     right_bound = 1250 - 10
#     top_bound = 10
#     bottom_bound = 900 - 10
#     toplx, toply = 12,15
#     toprx, topry = 1240, 15
#     bottomlx, bottomly = 15 , 894
#     bottomrx, bottomry = 1244 , 880


#     # tic = time.perf_counter()


#     dx, dy = command
#     canvas.move(car.shape, dx, dy)
#     for wheel in car.wheels:
#         canvas.move(wheel, dx, dy)
#     for sensor in car.sensors:
#         canvas.move(sensor, dx, dy)
#     car.x += dx
#     car.y += dy

def publish_controller_data( canvas, car, command,eat ,noeat):
    dx, dy, rotation, _, _ = command
    rad_angle = radians(car.angle)
    dx_rot = dx * cos(rad_angle) - dy * sin(rad_angle)
    dy_rot = dx * sin(rad_angle) + dy * cos(rad_angle)

    new_x = car.x + dx_rot
    new_y = car.y + dy_rot

    # Define the rectangle boundaries
    left_bound = 10
    right_bound = 1250 - 10
    top_bound = 10
    bottom_bound = 900 - 10

    if left_bound < new_x < right_bound and top_bound < new_y < bottom_bound:
        canvas.move(car.shape, dx_rot, dy_rot)
        for wheel in car.wheels:
            canvas.move(wheel, dx_rot, dy_rot)
        for sensor in car.sensors:
            canvas.move(sensor, dx_rot, dy_rot)
        car.x = new_x
        car.y = new_y

    if rotation != 0:
        car.is_rotating = True
        car.rotation_direction = 1 if rotation > 0 else -1
        canvas.after(200, lambda: stop_rotation(car))  # Schedule stop after 200ms



def move_to_targetv2(target_position, car, canvas, eat, noeat):
    # Initialize PID controllers
    Kp_angle, Ki_angle, Kd_angle = 0.001, 0, 0.05
    Kp_dist, Ki_dist, Kd_dist = 0.01, 0, 0.05
    angle_pid = PIDController(Kp_angle, Ki_angle, Kd_angle)
    dist_pid = PIDController(Kp_dist, Ki_dist, Kd_dist)
    
    # Commands
    comswallow = (0, 0, 0, 1, 0)

    # Get the project's root directory
    # project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # json_file_path = os.path.join(project_root, 'robot.json')

    # De-structure the target position
    target_x, target_y = target_position
    position_threshold = 185 #gammel 190
    angle_threshold = 11
    
    
    # Load car values into the car object
    # car = get_car_data_from_json(json_file_path)
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
        publish_controller_data(canvas, car,0,0,0)  # Activate intake at target
        # publish_controller_data(canvas, car,0,0,comswallow)  # Activate intake at target
        return 1


    # Angle correction
    print(f"angle error: {abs(angle_error)}\n")
    if abs(angle_error) > angle_threshold:
        angle_correction = angle_pid.calculate(0, angle_error)
        if angle_error > 180:#Der forventes at skulle skrues her: 

            publish_controller_data((canvas, car, max(0.12, min(angle_correction, 1)), 0, 0))  # Tilt right
        else:
            publish_controller_data((canvas, car, max(-0.12, min(angle_correction, -1)), 0, 0))  # Tilt left
        return 0
    
    # Forward movement control
    forward_speed = dist_pid.calculate(0, distance)
    forward_speed = max(0.15, min(forward_speed, 1))  # Clamp forward speed between 0.15 and 1
    publish_controller_data((canvas, car, forward_speed, 0, 0))  # Move forward
    return 0



#TRUE
# def rotate_car():
#     window = tk.Tk()
#     window.title("Car Rotation")
#     window.resizable(False,False)
#     canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
#     canvas.pack()
#     draw_rectangle(canvas)

#     car = Car(canvas, 75, 75, 0)
#     car.angle = 0
#     draw_car(canvas, car)

#     car.rotate()

#     car.is_rotating = False
#     coord_label = tk.Label(window, text="Car Coordinates: x=200, y=200")

#     targetX, targetY = 200, 300

#     canvas.create_oval(targetX - 10, targetY - 10, targetX + 10, targetY + 10, outline="black", width=2, fill='pink')

#     # Define the y-range for the green dots
#     green_dot_y_range = (car.y + 30, car.y + 75)
#     # animate_car(canvas, car, targetX, targetY, coord_label, green_dot_y_range)

#     coord_label.pack()
#     canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))
#     window.bind('<Key>', lambda event: key_handler(event, car, canvas))

#     window.mainloop()

def rotate_car():
    window = tk.Tk()
    window.title("Car Rotation")
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()
    draw_rectangle(canvas)

    car = Car(canvas, 75, 75, 0)
    car.angle = 0
    draw_car(canvas, car)

    coord_label = tk.Label(window, text="Car Coordinates: x=200, y=200")

    targetX, targetY = 200, 300

    canvas.create_oval(targetX - 10, targetY - 10, targetX + 10, targetY + 10, outline="black", width=2, fill='pink')

    animate_car(canvas, car, targetX, targetY, coord_label)

    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    window.mainloop()

rotate_car()
