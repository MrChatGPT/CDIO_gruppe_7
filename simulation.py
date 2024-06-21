import matplotlib.pyplot as plt
import numpy as np


def simulate(white_balls, orange_balls, cross, car, ball):
    plt.figure(figsize=(12.5, 9.5))
    plt.xlim(0, 1250)
    plt.ylim(950, 0)  # Invert the y-axis to have (0,0) at the top-left corner

    for wb in white_balls:
        plt.plot(wb.x, wb.y, 'ko')  # white dot
    for ob in orange_balls:
        plt.plot(ob.x, ob.y, 'yo')  # orange dot
    for arm in cross.arms:
        plt.plot([arm.start[0], arm.end[0]], [arm.start[1], arm.end[1]], 'go-')
    plt.plot(car.x, car.y, 'bo')
    print("Ball: ",ball)
    plt.plot(ball.x, ball.y, 'mo')
    if len(ball.waypoints) > 0:
        for waypoint in ball.waypoints:
            plt.plot(waypoint.x, waypoint.y, 'mo')
        
    car_angle_radians = np.radians(car.angle)
    vector_length = 50
    vector_end_x = car.x + vector_length * np.cos(car_angle_radians)
    vector_end_y = car.y + vector_length * np.sin(car_angle_radians)
    plt.arrow(car.x, car.y, vector_end_x - car.x, vector_end_y - car.y, head_width=10, head_length=15, fc='g', ec='g')


    plt.show()
