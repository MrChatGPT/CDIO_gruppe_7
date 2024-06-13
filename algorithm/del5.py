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
        self.turn = 1  # Set turn to a high value to rotate indefinitely
        self.shape = None
        self.wheels = []
        self.sensors = []
        self.tkimage = None
        self.canvas_obj = None
        self.rotation_direction = 1  # 1 for clockwise, -1 for counterclockwise
        self.is_rotating = False  # Flag to control rotation

    def rotate(self):
        self.angle = (self.angle + self.rotation_direction) % 360  # Change direction based on rotation_direction
        print(f"Current angle: {self.angle}")  # Print the current angle
        self.update_position()
        self.canvas.after(1000, self.rotate)

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

def move_to_target(car, target_position, green_dot_y_range, canvas):


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

    print(f"current x,y {current_x},{current_y}")

    if green_dot_y_range[0] <= current_y <= green_dot_y_range[1] and abs(dx) <= threshhold:
        car.is_rotating = False  # Stop rotating
        return comstop  # The car is within the range of the green dots and close to the target x
    

     # Calculate the angle to the target
    desired_angle = math.atan2(dy, dx) * (180 / math.pi)  # Convert radians to degrees
    if desired_angle < 0:
        desired_angle += 360

    current_angle = car.angle  # Assuming car.angle is in degrees

    print(f"Current position: ({current_x}, {current_y}), Target position: ({target_x}, {target_y})")
    print(f"Current angle: {current_angle}, Desired angle: {desired_angle}")

    if green_dot_y_range[0] <= current_y <= green_dot_y_range[1] and abs(dx) <= threshhold:
        car.is_rotating = False  # Stop rotating
        return comstop  # The car is within the range of the green dots and close to the target x
    
    angle_diff = (desired_angle - current_angle + 360) % 360
    if angle_diff > 180:
        angle_diff -= 360

    # Determine the movement based on the angle difference
    if abs(angle_diff) < 10:  # If the angle difference is small, move forward
        car.is_rotating = False
        return comforward if dx > 0 else comtiltleft
    else:
        car.is_rotating = True
        if angle_diff > 0:
            car.rotation_direction = 1
            return comtiltright  # Rotate clockwise
        else:
            car.rotation_direction = -1
            return comtiltleft  # Rotate anti-clockwise


    # if (current_x and target_x) == 1:
    #     angle = car.angle
    #     i

    
    # if abs(dx) > abs(dy):
    #     print(f"dx is bigger than dy")
    #     # Move in the x direction

    #     if dx > 0:
    #         car.rotation_direction = 1
    #         car.is_rotating = True  # Flag to control rotation
    #         print(f"rotate clockwise")
    #         return (5, 0)  # Move right
            
        
    #     else:
    #         car.rotation_direction = -1
    #         car.is_rotating = True  # Flag to control rotation
    #         print(f"rotate anti-clockwise")
    #         return (-5, 0)  # Move left

    # if current_y + 125 < target_y - threshhold:  # If our y is sharply less than the target y - 20
    #     if not((car.angle < 0 + threshhold) or (car.angle > 360 - threshhold)):  # Angle is not between 340 and 20 degrees
    #         print("desired angle: 340-20, actual:", car.angle)
    #         car.rotation_direction = 0
    #         car.is_rotating = True  # Start rotating
    #         # move_car(canvas, car, comtiltright)
    #         # time.sleep(0.2)
    #         return comtiltright
    #     print("We shall move forwards")
    #     car.rotation_direction = 0
    #     car.is_rotating = False  # Stop rotating
    #     # move_car(canvas, car, comforward)
    #     # time.sleep(0.2)
    #     return comforward
    # elif current_y - 125 > target_y + threshhold:  # If our y is sharply greater than the target y + 20
    #     if not((car.angle > 180 - threshhold) or (car.angle < 180 + threshhold)):  # Angle is not between 170 and 190 degrees
    #         print("desired angle: 170-190, actual:", car.angle)
    #         car.rotation_direction = 1
    #         car.is_rotating = True  # Start rotating
    #         # move_car(canvas, car, comtiltright)
    #         # time.sleep(0.2)
    #         car.is_rotating = False  # Stop rotating
    #         return comtiltright
    #     print("We should move forwards")
    #     car.rotation_direction = 0
    #     car.is_rotating = False  # Stop rotating
    #     # move_car(canvas, car, comforward)
    #     # time.sleep(0.2)
    #     return comforward

    # if current_x > target_x + threshhold:  # Move left
    #     if not((car.angle < 90 + threshhold) or (car.angle > 90 - threshhold)):  # Angle is not between 70 and 110 degrees
    #         car.rotation_direction = 1
    #         car.is_rotating = True  # Start rotating
    #         # move_car(canvas, car, comtiltleft)
    #         # time.sleep(0.2)
    #         car.is_rotating = False  # Stop rotating
    #         return comtiltleft
    #     car.rotation_direction = 0
    #     car.is_rotating = False  # Stop rotating
    #     # move_car(canvas, car, comforward)
    #     # time.sleep(0.2)
    #     return comforward
    # elif current_x < target_x - threshhold:  # Move right
    #     if not((car.angle < 270 + threshhold) or (car.angle > 270 - threshhold)):  # Angle is not between 250 and 290 degrees
    #         car.rotation_direction = 1
    #         car.is_rotating = True  # Start rotating
    #         # move_car(canvas, car, comtiltright)
    #         # time.sleep(0.2)
    #         car.is_rotating = False  # Stop rotating
    #         return comtiltright
    #     car.rotation_direction = 0
    #     car.is_rotating = False  # Stop rotating
    #     # move_car(canvas, car, comforward)
    #     # time.sleep(0.2)
    #     return comforward

    # return comstop



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
    command = move_to_target(car, (targetX, targetY), green_dot_y_range, canvas)
    move_car(canvas, car, command)
    coord_label.config(text=f"Car Coordinates: x={car.x}, y={car.y}, angle={car.angle}")
    # if command != (0, 0):
    #     canvas.after(250, animate_car, canvas, car, targetX, targetY, coord_label, green_dot_y_range)
    canvas.after(250, animate_car, canvas, car, targetX, targetY, coord_label, green_dot_y_range)


def rotate_car():
    window = tk.Tk()
    window.title("Car Rotation")
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()
    draw_rectangle(canvas)

    car = Car(canvas, 200, 200, 0)
    draw_car(canvas, car)

    car.rotate()

    coord_label = tk.Label(window, text="Car Coordinates: x=200, y=200")
    
    targetX, targetY = 800, 100
    canvas.create_oval(targetX-10, targetY-10, targetX+10, targetY+10, outline="black", width=2, fill='pink')
    
    # Define the y-range for the green dots
    green_dot_y_range = (car.y + 30, car.y + 75)
    animate_car(canvas, car, targetX, targetY, coord_label, green_dot_y_range)
    

    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    window.mainloop()

rotate_car()
