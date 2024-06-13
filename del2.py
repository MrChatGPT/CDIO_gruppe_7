import json
from time import sleep
import numpy as np
import os
import tkinter as tk

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

def move_to_target(car, target_position):
    current_x, current_y = car.x, car.y
    target_x, target_y = target_position

    comstop = (0, 0)
    comtiltleft = (0, -5)
    comtiltright = (0, 5)
    comforward = (5, 0)

    threshhold = 20

    # Calculate the differences
    dx = target_x - current_x
    dy = target_y - current_y

    if abs(dx) <= threshhold and abs(dy) <= threshhold:
        return comstop  # The car is within the threshold of the target

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
    beginx, beginy, carx, cary = car.x, car.y, car.x + 100, car.y + 100
    car.shape = canvas.create_rectangle(beginx, beginy, carx, cary, outline="black", width=2, fill='darkgrey')
    wheel_radius = 10
    wheel_positions = [
        (beginx - wheel_radius, beginy - wheel_radius, beginx + wheel_radius, beginy + wheel_radius),
        (carx - wheel_radius, beginy - wheel_radius, carx + wheel_radius, beginy + wheel_radius),
        (beginx - wheel_radius, cary - wheel_radius, beginx + wheel_radius, cary + wheel_radius),
        (carx - wheel_radius, cary - wheel_radius, carx + wheel_radius, cary + wheel_radius)
    ]
    
    car.wheels = [canvas.create_oval(pos, outline="black", width=2, fill='black') for pos in wheel_positions]
    sensor_positions = [(beginx + 95, beginy + 30), (beginx + 95, beginy + 75)]
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

def animate_car(canvas, car, targetX, targetY, coord_label):
    command = move_to_target(car, (targetX, targetY))
    move_car(canvas, car, command)
    coord_label.config(text=f"Car Coordinates: x={car.x}, y={car.y}")
    if command != (0, 0):
        canvas.after(100, animate_car, canvas, car, targetX, targetY, coord_label)

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
    animate_car(canvas, car, targetX, targetY, coord_label)

    window.mainloop()

runSim()

    
