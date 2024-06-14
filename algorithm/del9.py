from math import radians, cos, sin, pi
import math
import time
import tkinter as tk

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def calculate(self, setpoint, measured_value):
        error = setpoint - measured_value
        self.integral += error
        derivative = error - self.previous_error
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

def publish_controller_data(command, car, canvas):
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

def stop_rotation(car):
    car.is_rotating = False
    car.rotation_direction = 0

def move_to_targetv2(target_position, car, canvas):
    # Initialize PID controllers
    Kp_angle, Ki_angle, Kd_angle = 0.001, 0, 0.05
    Kp_dist, Ki_dist, Kd_dist = 0.01, 0, 0.05
    angle_pid = PIDController(Kp_angle, Ki_angle, Kd_angle)
    dist_pid = PIDController(Kp_dist, Ki_dist, Kd_dist)
    
    # Commands
    comswallow = (0, 0, 0, 1, 0)

    # De-structure the target position
    target_x, target_y = target_position
    position_threshold = 185
    angle_threshold = 11
    
    # Load car values into the car object
    current_x, current_y, current_angle = car.x, car.y, car.angle
    
    print(f"Desired position: {target_position}\nMy position: ({current_x}, {current_y}), angle: {current_angle}")
    
    # Calculate distance and desired angle
    distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
    
    desired_angle = (math.degrees(math.atan2(target_y - current_y, target_x - current_x))-90) % 360
    print(f"Desired angle:{desired_angle}\n")
    
    angle_error = (desired_angle - current_angle) % 360
    if angle_error > 180:
        angle_error -= 360
    
    if (distance < position_threshold) and (abs(angle_error) < angle_threshold):
        print("Target reached!")
        publish_controller_data(comswallow, car, canvas)  # Activate intake at target
        return 1

    # Angle correction
    print(f"angle error: {abs(angle_error)}\n")
    if abs(angle_error) > angle_threshold:
        angle_correction = angle_pid.calculate(0, angle_error)
        if angle_error > 180:
            publish_controller_data((0, 0, max(0.12, min(angle_correction, 1)), 0, 0), car, canvas)  # Tilt right
        else:
            publish_controller_data((0, 0, max(-0.12, min(angle_correction, -1)), 0, 0), car, canvas)  # Tilt left
        return 0
    
    # Forward movement control
    forward_speed = dist_pid.calculate(0, distance)
    forward_speed = max(0.15, min(forward_speed, 1))  # Clamp forward speed between 0.15 and 1
    publish_controller_data((0, forward_speed, 0, 0, 0), car, canvas)  # Move forward
    return 0

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

def animate_car(canvas, car, targetX, targetY, coord_label):
    if move_to_targetv2((targetX, targetY), car, canvas) == 0:
        coord_label.config(text=f"Car Coordinates: x={car.x}, y={car.y}, angle={car.angle}")
        canvas.after(100, animate_car, canvas, car, targetX, targetY, coord_label)

def rotate_car():
    window = tk.Tk()
    window.title("Car Rotation")
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()
    draw_rectangle(canvas)

    car = Car(canvas, 60, 60, 0)
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
