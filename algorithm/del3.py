import json
import numpy as np
import math
import os
import tkinter as tk
from math import radians, cos, sin
from random import randint

class Car:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

    def __repr__(self):
        return f"Car(x={self.x}, y={self.y}, angle={self.angle})"

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

def move_to_target(car, target_position, green_dot_y_range):
    current_x, current_y = car.x, car.y
    target_x, target_y = target_position
    target_x -= 50
    target_y -= 100

    comstop = (0, 0)
    comtiltleft = (0, -5)
    comtiltright = (0, 5)
    comforward = (5, 0)

    threshhold = 20

    # Calculate the differences
    dx = target_x - current_x
    dy = target_y - current_y

    # Calculate the angle to the target
    angle_rad = math.atan2(-dy, dx)  # Invert y to account for image coordinate system
    angle_deg = math.degrees(angle_rad) + 90
    if angle_deg < 0:
        angle_deg += 360
    angle_deg = int(round(angle_deg))

    # Update the car's angle
    car.angle = angle_deg
    print(f"car angle{angle_deg}")
    if green_dot_y_range[0] <= current_y <= green_dot_y_range[1] and abs(dx) <= threshhold:
        return comstop  # The car is within the range of the green dots and close to the target x

    if abs(dx) > abs(dy):
        # Move in the x direction
        if dx > 0:
            return (5, 0)  # Move right
        else:
            return (-5, 0)  # Move left
    else:
        # Move in the y direction
        if dy > 0:
            return (0, 5)  # Move down
        else:
            return (0, -5)  # Move up

def draw_rectangle(canvas):
    canvas.create_rectangle(10, 10, 1250, 900, outline="red", width=10)
    center_x = 1250 // 2
    center_y = 900 // 2
    cross_size = 40
    canvas.create_line(center_x - cross_size, center_y, center_x + cross_size, center_y, fill="red", width=5)
    canvas.create_line(center_x, center_y - cross_size, center_x, center_y + cross_size, fill="red", width=5)

def draw_car(canvas, car):
    beginx, beginy, carx, cary = car.x, car.y, car.x, car.y
    car.shape = canvas.create_polygon(beginx, beginy, carx, cary, outline="black", width=2, fill='darkgrey')
    wheel_radius = 10
    wheel_positions = [
        (beginx - wheel_radius, beginy - wheel_radius, beginx + wheel_radius, beginy + wheel_radius),
        (carx - wheel_radius, beginy - wheel_radius, carx + wheel_radius, beginy + wheel_radius),
        (beginx - wheel_radius, cary - wheel_radius, beginx + wheel_radius, cary + wheel_radius),
        (carx - wheel_radius, cary - wheel_radius, carx + wheel_radius, cary + wheel_radius)
    ]
    
    car.wheels = [canvas.create_oval(pos, outline="black", width=2, fill='black') for pos in wheel_positions]
    sensor_positions = [(beginx + 30, beginy + 95), (beginx + 75, beginy + 95)]
    car.sensors = [canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="black", width=2, fill='green') for x, y in sensor_positions]

def move_car(canvas, car, command):
    dx, dy = command
    canvas.move(car.shape, dx, dy)
    for wheel in car.wheels:
        canvas.move(wheel, dx, dy)
    for sensor in car.sensors:
        canvas.move(sensor, dx, dy)
    car.x += dx
    car.y += dy

def animate_car(canvas, car, targetX, targetY, coord_label, green_dot_y_range):
    command = move_to_target(car, (targetX, targetY), green_dot_y_range)
    move_car(canvas, car, command)
    coord_label.config(text=f"Car Coordinates: x={car.x}, y={car.y}, angle={car.angle}")
    if command != (0, 0):
        canvas.after(100, animate_car, canvas, car, targetX, targetY, coord_label, green_dot_y_range)

def update_mouse_coordinates(event, coord_label):
    coord_label.config(text=f"Mouse Coordinates: x={event.x}, y={event.y}")

def runSim():
    window = tk.Tk()
    window.title("Rectangle Drawing")
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()

    draw_rectangle(canvas)
    
    car = Car(100, 100, 0)
    draw_car(canvas, car)

    coord_label = tk.Label(window, text="Car Coordinates: x=100, y=100")
    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    targetX, targetY = 800, 600
    canvas.create_oval(targetX-10, targetY-10, targetX+10, targetY+10, outline="black", width=2, fill='pink')
    
    # Define the y-range for the green dots
    green_dot_y_range = (car.y + 30, car.y + 75)
    
    animate_car(canvas, car, targetX, targetY, coord_label, green_dot_y_range)

    window.mainloop()


#########################################
# runSim()


##########################################

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

    def rotate(self):
        if self.turn > 0:
            self.angle = (self.angle - 1) % 360
            self.turn -= 1
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

def rotate_car():
    window = tk.Tk()
    window.title("Car Rotation")
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()

    car = Car(canvas, 200, 200, 0)
    draw_car(canvas, car)

    car.turn = randint(30, 360)
    car.rotate()

    coord_label = tk.Label(window, text="Car Coordinates: x=200, y=200")
    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    window.mainloop()

rotate_car()