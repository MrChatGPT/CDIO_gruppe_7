import tkinter as tk

def draw_rectangle(canvas):
    # Draw a red rectangle
    canvas.create_rectangle(10, 10, 1250, 900, outline="red", width=10)

    # Calculate the center of the canvas
    center_x = 1250 // 2
    center_y = 900 // 2

    # Draw a centered red cross
    cross_size = 40  # size of the cross
    canvas.create_line(center_x - cross_size, center_y, center_x + cross_size, center_y, fill="red", width=5)
    canvas.create_line(center_x, center_y - cross_size, center_x, center_y + cross_size, fill="red", width=5)

def draw_car(canvas, car):
    beginx, beginy, carx, cary = car['position']
    # Draw the base of the car
    car['base'] = canvas.create_rectangle(beginx, beginy, carx, cary, outline="black", width=2, fill='darkgrey')
    
    # Draw the wheels
    wheel_radius = 10
    wheel_positions = [
        (beginx - wheel_radius, beginy - wheel_radius, beginx + wheel_radius, beginy + wheel_radius), # Top-left wheel
        (carx - wheel_radius, beginy - wheel_radius, carx + wheel_radius, beginy + wheel_radius), # Top-right wheel
        (beginx - wheel_radius, cary - wheel_radius, beginx + wheel_radius, cary + wheel_radius), # Bottom-left wheel
        (carx - wheel_radius, cary - wheel_radius, carx + wheel_radius, cary + wheel_radius)  # Bottom-right wheel
    ]
    
    car['wheels'] = [canvas.create_oval(pos, outline="black", width=2, fill='black') for pos in wheel_positions]

    # Draw some small circles (representing sensors or other components)
    sensor_positions = [(beginx + 95, beginy + 30), (beginx + 95, beginy + 75)]
    car['sensors'] = [canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="black", width=2, fill='green') for x, y in sensor_positions]

def move_car(canvas, car, dx, dy):
    # Move the base
    canvas.move(car['base'], dx, dy)
    # Move the wheels
    for wheel in car['wheels']:
        canvas.move(wheel, dx, dy)
    # Move the sensors
    for sensor in car['sensors']:
        canvas.move(sensor, dx, dy)
    # Update car's position
    car['position'] = (car['position'][0] + dx, car['position'][1] + dy, car['position'][2] + dx, car['position'][3] + dy)

def animate_car(canvas, car, targetX, targetY, coord_label):
    currentX = car['position'][0]
    currentY = car['position'][1]
    car_width = car['position'][2] - car['position'][0]
    car_height = car['position'][3] - car['position'][1]
    sensor_offset_x = 95
    sensor_offset_y1 = 30
    sensor_offset_y2 = 75
    
    sensor1_x = currentX + sensor_offset_x
    sensor1_y = currentY + sensor_offset_y1
    sensor2_x = currentX + sensor_offset_x
    sensor2_y = currentY + sensor_offset_y2
    
    # Calculate the distance to move
    dx = 5 if sensor1_x < targetX else -5
    dy = 5 if sensor1_y < targetY else -5
    
    if (abs(sensor1_x - targetX) > abs(dx)) or (abs(sensor1_y - targetY) > abs(dy)):
        move_car(canvas, car, dx, dy)
        # Update the coordinates label
        coord_label.config(text=f"Car Coordinates: x={car['position'][0]}, y={car['position'][1]}")
        canvas.after(100, animate_car, canvas, car, targetX, targetY, coord_label)
    else:
        # Move the car to the exact target position if close enough
        final_dx = targetX - sensor1_x
        final_dy = targetY - sensor1_y
        move_car(canvas, car, final_dx, final_dy)
        coord_label.config(text=f"Car Coordinates: x={car['position'][0]}, y={car['position'][1]}")

def update_mouse_coordinates(event, coord_label):
    coord_label.config(text=f"Mouse Coordinates: x={event.x}, y={event.y}")

def runSim():
    # Create a new window
    window = tk.Tk()
    window.title("Rectangle Drawing")

    # Create a canvas widget 1250, 900
    canvas = tk.Canvas(window, width=1260, height=910, bg='lightgrey')
    canvas.pack()

    draw_rectangle(canvas)
    
    car = {'position': (100, 100, 200, 200)}
    draw_car(canvas, car)

    # Create a label to display the car coordinates
    coord_label = tk.Label(window, text="Car Coordinates: x=100, y=100")
    coord_label.pack()

    # Bind mouse motion to update coordinates
    canvas.bind('<Motion>', lambda event: update_mouse_coordinates(event, coord_label))

    # Start the animation towards a target position (e.g., 800, 600)
    targetX, targetY = 800, 600
    canvas.create_oval(targetX-10, targetY-10, targetX+10, targetY+10, outline="black", width=2, fill='pink')
    animate_car(canvas, car, targetX, targetY, coord_label)

    # Run the Tkinter event loop
    window.mainloop()

########################################################################################
runSim()
