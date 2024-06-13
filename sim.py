import tkinter as tk

def draw_rectangle(canvas):
    # Draw a red rectangle
    canvas.create_rectangle(50, 50, 1200, 850, outline="red", width=4)

    # Calculate the center of the canvas
    center_x = 1250 // 2
    center_y = 900 // 2

    # Draw a centered red cross
    cross_size = 40  # size of the cross
    canvas.create_line(center_x - cross_size, center_y, center_x + cross_size, center_y, fill="red", width=4)
    canvas.create_line(center_x, center_y - cross_size, center_x, center_y + cross_size, fill="red", width=4)

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

def animate_car(canvas, car, targetX, targetY):
    currentX = car['position'][0]
    currentY = car['position'][1]
    
    # Calculate the distance to move
    dx = 5 if currentX < targetX else -5
    dy = 5 if currentY < targetY else -5
    
    if (abs(currentX - targetX) > abs(dx)) or (abs(currentY - targetY) > abs(dy)):
        move_car(canvas, car, dx, dy)
        canvas.after(100, animate_car, canvas, car, targetX, targetY)
    else:
        # Move the car to the exact target position if close enough
        move_car(canvas, car, targetX - currentX, targetY - currentY)

def runSim():
    # Create a new window
    window = tk.Tk()
    window.title("Rectangle Drawing")

    # Create a canvas widget 1250, 900
    canvas = tk.Canvas(window, width=1250, height=900, bg='lightgrey')
    canvas.pack()

    draw_rectangle(canvas)
    
    car = {'position': (100, 100, 200, 200)}
    draw_car(canvas, car)

    # Start the animation towards a target position (e.g., 800, 600)
    targetX, targetY = 800, 600
    animate_car(canvas, car, targetX, targetY)

    # Run the Tkinter event loop
    window.mainloop()

########################################################################################
runSim()
