import tkinter as tk
from math import radians, cos, sin
from random import randint

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

def rotate_car():
    window = tk.Tk()
    window.title("Car Rotation")
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()

    car = Car(canvas, 200, 200, 0)
    draw_car(canvas, car)

    car.rotate()

    coord_label = tk.Label(window, text="Car Coordinates: x=200, y=200")
    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    # Add buttons to change rotation direction
    def set_rotation_direction(direction):
        car.rotation_direction = direction

    button_frame = tk.Frame(window)
    button_frame.pack()

    clockwise_button = tk.Button(button_frame, text="Clockwise", command=lambda: set_rotation_direction(1))
    clockwise_button.pack(side=tk.LEFT)

    counterclockwise_button = tk.Button(button_frame, text="Counterclockwise", command=lambda: set_rotation_direction(-1))
    counterclockwise_button.pack(side=tk.LEFT)

    window.mainloop()

rotate_car()
