import json
from time import sleep
import numpy as np 
import os


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
def LoadOrangeBall(filename="orangeballs.json"):
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


# Function to calculate the distance between the Robot and the balls
# This function is based on the dictance formula: sqrt((x2-x1)^2 +(y2-y1)^2)

def Distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to sort the positions of the balls based on their distance from the Robot 
# This function is based on the key function lambda, where the ist will be sorted in descending order

def SortByDistance(RobotXY=LoadRobot(), BallsXY=LoadBalls(), OrangeBallXY = LoadOrangeBall() ):
       
    # Now we sort the balls based on their distance to the Robot
    # Here we use the lam
    SortedList = sorted(BallsXY, key=lambda ball: Distance(RobotXY, ball))
    
    # Add the orange ball at the end
    SortedList.append(OrangeBallXY)
    print(SortedList[0])
    return SortedList[0]




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
