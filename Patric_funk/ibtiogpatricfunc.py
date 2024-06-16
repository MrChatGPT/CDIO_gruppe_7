# publish_controller_data(0.1, 0.1,   0.1,   0,    0)
#                          x,    y,   phi,  eat,  eject
#publish_controller_data(0.1,0.1,0.1,0,0)
import json
from time import sleep
import numpy as np 
import os

class Car:
    def __init__(self, x, y, angle):
        self.x = x #car.x
        self.y = y #car.y
        self.angle = angle #car.angle

    def __repr__(self): #hvis man skriver print(car), så:
        return f"Car(x={self.x}, y={self.y}, angle={self.angle})"

def get_car_data_from_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Assuming the JSON structure is as mentioned: [[0, 32, 0]]
    if data and isinstance(data, list) and len(data) > 0:
        car_data = data[0]
        if len(car_data) == 3:
            x, y, angle = car_data
            return Car(x, y, angle)
        else:
            raise ValueError("Invalid car data structure in JSON file.")
    else:
        raise ValueError("Invalid JSON structure.")

def move_to_target(target_position):
    # load car values into the car object
    # Get the project's root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Construct the path to the robot.json file in the root directory
    json_file_path = os.path.join(project_root, 'robot.json')
    car = get_car_data_from_json(json_file_path)
    print(car)
    # Extract the current position from the car object
    current_x, current_y = car.x, car.y
    
    # De-structure the target position
    target_x, target_y = target_position
    #print(f"Target_x = {target_x}\nTarget_y = {target_y}")

    #Commands
    comstop = (0,0,0,0,0)
    comtiltleft = (0,0,-0.15,0,0)
    comtiltright = (0,0,0.15,0,0)
    comforward = (0,0.15,0,0,0)

    # Move in the y direction
    if current_y != target_y:
        if current_y < target_y: #vi skal altså op ad ^
            #overvej at smid et threshold ind her
            if car.angle != 0:
                if car.angle > 0:
                    publish_controller_data(comtiltleft) #vi vender os mod venstre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
                if car.angle > 270:
                    publish_controller_data(comtiltright) #vi vender os mod højre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
            publish_controller_data(comforward) #vi rykker 0.15 i y-retningen (opad)
            sleep(0.5)
            publish_controller_data(comstop)
            return
        else: #så skal vi altså nedad ˅
            if car.angle != 180:
                if car.angle > 0:
                    publish_controller_data(comtiltright) #vi vender tilføjer grader indtil vi er på 180
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
                if car.angle > 180:
                    publish_controller_data(comtiltleft)#vi "fjerner" grader indtil vi er på 180
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
            publish_controller_data(comforward) #vi rykker 0.15 i y-retningen (opad)
            sleep(0.5)
            publish_controller_data(comstop)
            return
    
    # Move in the x direction
    # nu er vores angle enten 0 eller 180:
    if current_x != target_x:
        if current_x > target_x: #så skal vi til venstre
            if car.angle != 270:
                if car.angle == 0:
                    publish_controller_data(comtiltleft) #hvis vi peger opad ved 0 grader, skal vi bare tilte mod venstre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
                if car.angle > 270: #efter at have trukket én fra 0, lander vi på 359 grader
                    publish_controller_data(comtiltleft) #Hvis vi er større end 270, skal vi bare blive ved med at tilte mod venstre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
                if car.angle == 180:
                    publish_controller_data(comtiltright) #hvis vi allerede er ved 180, skal vi bare mod højre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return    
                if car.angle < 270:
                    publish_controller_data(comtiltright) #Hvis vi er større end 180, skal vi fortsat mod højre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
            publish_controller_data(comforward)
            sleep(0.5)
            publish_controller_data(comstop)
            return
        else: #så skal vi til højre
            if car.angle != 90:
                if car.angle == 180:
                    publish_controller_data(comtiltleft) #hvis vi peger med næsen nedad, skal vi tilte med bilens venstre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return    
                if car.angle > 90:
                    publish_controller_data(comtiltleft) #hvis vi peger med næsen nedad, skal vi tilte med bilens venstre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return    
                if car.angle == 0:
                    publish_controller_data(comtiltright) #hvis vi peger med næsen opad, skal vi tilte mod bilens højre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
                if car.angle < 90:
                    publish_controller_data(comtiltright) #hvis vi peger med næsen opad, skal vi tilte mod bilens højre
                    sleep(0.5)
                    publish_controller_data(comstop)
                    return
            publish_controller_data(comforward)
            sleep(0.5)
            publish_controller_data(comstop)
            return
    #nu er current_y = target_y, current_x = target_x
    publish_controller_data(0,0,0,1,0) #og så skal der nedsvælges
    sleep(0.5)
    publish_controller_data(comstop)


# Function to read ball positions from a JSON file
def LoadBalls(filename="balls.json"):
    # Get the project's root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Construct the path to the robot.json file in the root directory
    json_file_path = os.path.join(project_root, filename)
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    # Convert the list of lists back to a list of tuples
    BallsXY = [tuple(center) for center in data]
    
    return BallsXY


# Function to read ball positions from a JSON file
def LoadOrangeBall(filename="orangeball.json"):
    # Get the project's root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Construct the path to the robot.json file in the root directory
    json_file_path = os.path.join(project_root, filename)
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the first element and convert it to a tuple
    if data and isinstance(data, list) and len(data) > 0:
        OrangeBallXY = tuple(data[0])
    else:
        raise ValueError("Invalid JSON structure or data not found.")
    
    return OrangeBallXY

# Function to read ball positions from a JSON file
def LoadRobot(filename="robot.json"):
    # Get the project's root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Construct the path to the robot.json file in the root directory
    json_file_path = os.path.join(project_root, filename)
    
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    # Extract the first two values from the first element in the list
    if data and isinstance(data, list) and len(data) > 0:
        RobotXY = tuple(data[0][:2])
    else:
        raise ValueError("Invalid JSON structure or data not found.")
    
    return RobotXY

# Loading the positions from the JSON file
BallsXY = LoadBalls()
OrangeBallXY = LoadOrangeBall() 
RobotXY = LoadRobot() 


# Function to calculate the distance between the Robot and the balls
# This function is based on the dictance formula: sqrt((x2-x1)^2 +(y2-y1)^2)

def Distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to sort the positions of the balls based on their distance from the Robot 
# This function is based on the key function lambda, where the ist will be sorted in descending order

def SortByDistance(RobotXY, BallsXY):
       
    # Now we sort the balls based on their distance to the Robot
    # Here we use the lam
    SortedList = sorted(BallsXY, key=lambda ball: Distance(RobotXY, ball))
    
    # Add the orange ball at the end
    SortedList.append(OrangeBallXY)
    
    return SortedList


    # The values of the list
#     SortedList = [
#     (2, 3),     # Closest (Distance: 3.61)
#     (1, 4),     # (Distance: 4.12)
#     (5, 1),     # (Distance: 5.10)
#     (6, 2),     # (Distance: 6.32)
#     (4, 6),     # (Distance: 7.21)
#     (3, 7),     # (Distance: 7.62)
#     (8, 3),     # (Distance: 8.54)
#     (7, 5),     # (Distance: 8.60)
#     (9, 1),     # Furthest (Distance: 9.06)
#     (7, 8)      # Orange Ball added at the end
# ]


# Example to pass the Ball position as a parameter
def ProcessFirstBall(BallPosition):
    print(f"This Ball position will be used to control the robot: {BallPosition}")
   


# Example of sorted list of balls by their distance to the Robot 
print("Unsorted list:", BallsXY)
SortedExample = SortByDistance(RobotXY, BallsXY.copy())
print("Sorted list:", SortedExample)

# Testing function parameter
FirstBallPosition = SortedExample[0]
#ProcessFirstBall(FirstBallPosition)

# Patric's function
move_to_target(FirstBallPosition)