import tkinter as tk
from math import radians, cos, sin
from random import randint

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
            (self.x - half_length - 10, self.y - half_width - 10),
            (self.x + half_length - 10, self.y - half_width - 10),
            (self.x - half_length - 10, self.y + half_width - 10),
            (self.x + half_length - 10, self.y + half_width - 10)
        ]
        rotated_wheels = [(self.x + (x - self.x) * cos_a - (y - self.y) * sin_a,
                           self.y + (x - self.x) * sin_a + (y - self.y) * cos_a)
                          for x, y in wheel_positions]
        for wheel, pos in zip(self.wheels, rotated_wheels):
            self.canvas.coords(wheel, pos[0] - 10, pos[1] - 10, pos[0] + 10, pos[1] + 10)

        # Calculate new coordinates for the sensors
        sensor_positions = [(self.x + 30, self.y + 45), (self.x + 75, self.y + 45)]
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

#   beginx, beginy, carx, cary = car.x, car.y, car.x + 100, car.y + 100
#     car.shape = canvas.create_rectangle(beginx, beginy, carx, cary, outline="black", width=2, fill='darkgrey')
#     wheel_radius = 10
#     wheel_positions = [
#         (beginx - wheel_radius, beginy - wheel_radius, beginx + wheel_radius, beginy + wheel_radius),
#         (carx - wheel_radius, beginy - wheel_radius, carx + wheel_radius, beginy + wheel_radius),
#         (beginx - wheel_radius, cary - wheel_radius, beginx + wheel_radius, cary + wheel_radius),
#         (carx - wheel_radius, cary - wheel_radius, carx + wheel_radius, cary + wheel_radius)
#     ]
    
#     car.wheels = [canvas.create_oval(pos, outline="black", width=2, fill='black') for pos in wheel_positions]
#     sensor_positions = [(beginx + 30, beginy + 95), (beginx + 75, beginy + 95)]
#     car.sensors = [canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="black", width=2, fill='green') for x, y in sensor_positions]


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
