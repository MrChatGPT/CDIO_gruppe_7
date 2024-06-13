import tkinter as tk
import json
import os

from algorithm.move_to_target import Car, get_car_data_from_json, move_to_target

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
    
    # Assuming the initial car position and angle
    car = Car(100, 100, 0)
    draw_car(canvas, car)

    coord_label = tk.Label(window, text="Car Coordinates: x=100, y=100")
    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    targetX, targetY = 800, 600
    canvas.create_oval(targetX-10, targetY-10, targetX+10, targetY+10, outline="black", width=2, fill='pink')
    animate_car(canvas, car, targetX, targetY, coord_label)

    # window.mainloop()

runSim()
import tkinter as tk
import json
import os

from algorithm.move_to_target import Car, get_car_data_from_json, move_to_target

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
    
    # Assuming the initial car position and angle
    car = Car(100, 100, 0)
    draw_car(canvas, car)

    coord_label = tk.Label(window, text="Car Coordinates: x=100, y=100")
    coord_label.pack()
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    targetX, targetY = 800, 600
    canvas.create_oval(targetX-10, targetY-10, targetX+10, targetY+10, outline="black", width=2, fill='pink')
    animate_car(canvas, car, targetX, targetY, coord_label)

    # window.mainloop()

runSim()
