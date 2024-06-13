import tkinter as tk

def draw_rectangle(canvas):
    # # Create a new window
    # window = tk.Tk()
    # window.title("Rectangle Drawing")

    # # Create a canvas widget 1250, 900
    # canvas = tk.Canvas(window, width=1250, height=900, bg='grey')
    # canvas.pack()

    # Draw a red rectangle
    canvas.create_rectangle(50, 50, 1200, 850, outline="red", width=4)

    # Calculate the center of the canvas
    center_x = 1250 // 2
    center_y = 900 // 2

    # Draw a centered red cross
    cross_size = 40  # size of the cross
    canvas.create_line(center_x - cross_size, center_y, center_x + cross_size, center_y, fill="red", width=4)
    canvas.create_line(center_x, center_y - cross_size, center_x, center_y + cross_size, fill="red", width=4)


    # #Simulate small white circles (stars) within the rectangle
    # stars = [(100, 100), (150, 120), (200, 140), (250, 160), (300, 180), (350, 200)]
    # for star in stars:
    #     canvas.create_oval(star[0], star[1], star[0] + 5, star[1] + 5, fill="white")
    # # Draw additional elements similar to the ones in the image
    # canvas.create_oval(330, 220, 340, 230, fill="red")
    # canvas.create_oval(350, 240, 370, 260, fill="red")

    # # Run the Tkinter event loop
    # window.mainloop()

# Call the function to draw the rectangle
# draw_rectangle()


def draw_car(canvas, car):
    # Create a new window
    # window = tk.Tk()
    # window.title("Robot Car Simulation")

    # # Create a canvas widget
    # canvas = tk.Canvas(window, width=800, height=600, bg='lightgrey')
    # canvas.pack()

    # beginx = 100
    # beginy = 100
    # carx = 200
    # cary = 200
    # # Draw the base of the car
    # canvas.create_rectangle(beginx, beginy, carx, cary, outline="black", width=2, fill='darkgrey')
   

    beginx, beginy, carx, cary = car['position']
    # Draw the base of the car
    car['base'] = canvas.create_rectangle(beginx, beginy, carx, cary, outline="black", width=2, fill='darkgrey')
    

    # Draw the wheels
    wheel_radius = 10
    wheel_positions = [
        (100 - wheel_radius, 100 - wheel_radius, 100 + wheel_radius, 100 + wheel_radius), # Top-left wheel
        (200 - wheel_radius, 100 - wheel_radius, 200 + wheel_radius, 100 + wheel_radius), # Top-right wheel
        (100 - wheel_radius, 200 - wheel_radius, 100 + wheel_radius, 200 + wheel_radius), # Bottom-left wheel
        (200 - wheel_radius, 200 - wheel_radius, 200 + wheel_radius, 200 + wheel_radius)  # Bottom-right wheel
    ]
    
    for pos in wheel_positions:
        canvas.create_oval(pos, outline="black", width=2, fill='black')

    # Draw some components on top of the car
    # canvas.create_rectangle(350, 250, 450, 300, outline="black", width=2, fill='green') # A green component
    # canvas.create_rectangle(300, 300, 400, 350, outline="black", width=2, fill='red')   # A red component
    # canvas.create_line(400, 200, 400, 400, fill="black", width=2) # Some wires or connectors
    # canvas.create_line(300, 200, 300, 400, fill="black", width=2)

    # # Draw some small circles
    sensor_positions = [(195, 130), (195, 175)]
    for (x, y) in sensor_positions:
        canvas.create_oval(x - 5, y - 5, x + 5, y + 5, outline="black", width=2, fill='green')







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


    # draw_car(canvas)

    # Run the Tkinter event loop
    window.mainloop()



########################################################################################
runSim()