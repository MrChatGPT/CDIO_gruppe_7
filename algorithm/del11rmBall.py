import tkinter as tk
from math import radians, cos, sin, atan2, pi
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
        if self.is_rotating:
            self.angle = (self.angle + self.rotation_direction) % 360  # Change direction based on rotation_direction
            print(f"Rotating: Current angle: {self.angle}")  # Print the current angle
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
            (self.x - half_length, self.y - half_width),
            (self.x + half_length, self.y - half_width),
            (self.x - half_length, self.y + half_width),
            (self.x + half_length, self.y + half_width)
        ]
        rotated_wheels = [(self.x + (x - self.x) * cos_a - (y - self.y) * sin_a,
                           self.y + (x - self.x) * sin_a + (y - self.y) * cos_a)
                          for x, y in wheel_positions]
        for wheel, pos in zip(self.wheels, rotated_wheels):
            self.canvas.coords(wheel, pos[0] - 10, pos[1] - 10, pos[0] + 10, pos[1] + 10)

        # Calculate new coordinates for the sensors
        sensor_positions = [(self.x - 10, self.y + 40), (self.x + 15, self.y + 40)]
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
    print(f"in draw car")  # Debug rotation direction
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

def draw_rectangle(canvas):
    canvas.create_rectangle(10, 10, 1250, 900, outline="red", width=10)
    center_x = 1250 // 2
    center_y = 900 // 2
    cross_size = 40
    canvas.create_line(center_x - cross_size, center_y, center_x + cross_size, center_y, fill="red", width=5)
    canvas.create_line(center_x, center_y - cross_size, center_x, center_y + cross_size, fill="red", width=5)

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
        print(f"Rotating with direction: {car.rotation_direction}")  # Debug rotation direction
        canvas.after(200, lambda: stop_rotation(car))  # Schedule stop after 200ms

def stop_rotation(car):
    car.is_rotating = False
    car.rotation_direction = 0

def move_to_targetv2(target_position, car, canvas, ball):
    # Initialize PID controllers
    Kp_angle, Ki_angle, Kd_angle = 0.001, 0, 0.05
    Kp_dist, Ki_dist, Kd_dist = 0.01, 0, 0.05
    angle_pid = PIDController(Kp_angle, Ki_angle, Kd_angle)
    dist_pid = PIDController(Kp_dist, Ki_dist, Kd_dist)
    
    # Commands
    comswallow = (0, 0, 0, 1, 0)
    print(f"in move to target v2")  # Debug rotation direction
    # De-structure the target position
    target_x, target_y = target_position
    position_threshold = 70 #185
    angle_threshold = 11
    
    # Load car values into the car object
    current_x, current_y, current_angle = car.x, car.y, car.angle
    
    print(f"Desired position: {target_position}\nMy position: ({current_x}, {current_y}), angle: {current_angle}")
    
    # Calculate distance and desired angle
    distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
    desired_angle = (math.degrees(atan2(target_y - current_y, target_x - current_x)) - 90) % 360
    print(f"Desired angle: {desired_angle}\n")
    
    angle_error = (desired_angle - current_angle) % 360
    if angle_error > 180:
        angle_error -= 360
    
    if (distance < position_threshold) and (abs(angle_error) < angle_threshold):
        print("Target reached!")
        canvas.after(100, lambda: delete_ball(canvas, ball))  
        # canvas.delete(ball)
        publish_controller_data((0, 0, 0, 1, 0), car, canvas)  # Activate intake at target
        return 1

    # Angle correction
    print(f"Angle error: {abs(angle_error)}\n")
    if abs(angle_error) > angle_threshold:
        angle_correction = angle_pid.calculate(0, angle_error)
        if angle_error > 0:
            publish_controller_data((0, 0, max(10.12, min(angle_correction, 1)), 0, 0), car, canvas)  # Tilt right
        else:
            publish_controller_data((0, 0, max(-10.12, min(angle_correction, -1)), 0, 0), car, canvas)  # Tilt left
        return 0
    
    # Forward movement control
    forward_speed = dist_pid.calculate(0, distance)
    forward_speed = max(10.15, min(forward_speed, 1))  # Clamp forward speed between 0.15 and 1
    print(f"Forward speed: {forward_speed}")  # Debug forward speed
    publish_controller_data((0, forward_speed, 0, 0, 0), car, canvas)  # Move forward
    return 0

def animate_car(canvas, car, targetX, targetY, coord_label, ball):
    if move_to_targetv2((targetX, targetY), car, canvas, ball) == 0:
        coord_label.config(text=f"Car Coordinates: x={car.x}, y={car.y}, angle={car.angle}")
        canvas.after(100, animate_car, canvas, car, targetX, targetY, coord_label, ball)



def draw_ball(canvas, targetX, targetY):
 
    ball = canvas.create_oval(targetX - 10, targetY - 10, targetX + 10, targetY + 10, outline="black", width=2, fill='pink')
  
    # Return the ids
    return ball



def delete_ball(canvas, ball):
    canvas.delete(ball)
    return





###Where the program begins###
def rotate_car():
    window = tk.Tk()
    window.title("Car Rotation")
    window.resizable(False,False)
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()
    draw_rectangle(canvas)

    car = Car(canvas, 75, 75, 0)
    car.angle = 0
    draw_car(canvas, car)

    car.rotate()

    coord_label = tk.Label(window, text="Car Coordinates: x=200, y=200")

    targetX, targetY = 1200, 200  # Changed target coordinates for better visibility

    # canvas.create_oval(targetX - 10, targetY - 10, targetX + 10, targetY + 10, outline="black", width=2, fill='pink')
  
    ball = draw_ball(canvas, targetX, targetY)
    animate_car(canvas, car, targetX, targetY, coord_label, ball)

    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    window.mainloop()

rotate_car()
