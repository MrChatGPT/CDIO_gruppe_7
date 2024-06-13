import json
import numpy as np
import matplotlib.pyplot as plt



# #******************************* FOR WINDOWS ******************************* 

# # Function to read coordinates from a json file
# def LoadCoordinatesWindows(filename):
#     # Get the project's root directory
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

#     # Construct the path to the robot.json file in the root directory
#     json_file_path = os.path.join(project_root, filename)
    
#     with open(json_file_path, 'r') as file:
#         data = json.load(file)
        
#     # Convert the list of lists back to a list of tuples
#     coordinates = [tuple(center) for center in data]
#     return coordinates


# # Function to read obstacle coordinates from a json file
# def LoadObstaclesWindows(filename="no_go_zones.json"):
#     # Get the project's root directory
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

#     # Construct the path to the robot.json file in the root directory
#     json_file_path = os.path.join(project_root, filename)

#     with open(filename, 'r') as file:
#         data = json.load(file)
        
#     # Extract the coordinates
#     obstacles = data[0]
#     return obstacles

# # Load the coordinates from the JSON files
# BallsXY = LoadCoordinatesWindows("whiteballs.json")
# OrangeBallXY = LoadCoordinatesWindows("orangeballs.json")[0]  # Only one orange ball, therefore take only the first element
# RobotXY = LoadCoordinatesWindows("robot.json")[0][:2]  # Only one robot, therefore take only the first element (without the angle)
# Obstacles = LoadObstaclesWindows()

# #******************************* FOR WINDOWS ******************************* 



#********************************* FOR IBTI ******************************** 

# Function to read coordinates from a json file
def LoadCoordinates(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    # Convert the nested list of lists back to a list of tuples
    coordinates = [tuple(center) for center in data]
    return coordinates


# Function to read obstacle coordinates from a json file
def LoadObstacles(filename="no_go_zones.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    # Extract the coordinates
    obstacles = data[0]
    return obstacles


# Load the coordinates from the json files
BallsXY = LoadCoordinates("whiteballs.json")
OrangeBallXY = LoadCoordinates("orangeballs.json")[0]  # Only one orange ball, take the first element
RobotXY = LoadCoordinates("robot.json")[0][:2]  # Only one robot, take the first element
ObstacleXY = LoadObstacles("no_go_zones.json")

#********************************* FOR IBTI ******************************** 



# #******************************* FOR DEBUGGING ******************************

# # Visualize the cross based on the coordinates
# def visualize_cross(obstacle):
#     # Extract the coordinates for plotting
#     x1, y1 = zip(*obstacle[0])
#     x2, y2 = zip(*obstacle[1])

#     # Plot the obstacle lines
#     plt.figure(figsize=(10, 10))

#     # Plot the diagonal lines of the cross
#     plt.plot(x1, y1, 'r-', linewidth=5, label='Diagonal Line 1')
#     plt.plot(x2, y2, 'b-', linewidth=5, label='Diagonal Line 2')

#     # Set the limits of the plot
#     plt.xlim(450, 650)
#     plt.ylim(350, 550)

#     # Adding labels and title
#     plt.xlabel('X Coordinate')
#     plt.ylabel('Y Coordinate')

#     # Add a legend
#     plt.legend()

#     # Display the plot
#     plt.grid(True)
#     plt.show()

# visualize_cross(obstacle)

# #******************************* FOR DEBUGGING ******************************



# Function to calculate the distance between the Robot and the balls
# This function is based on the Euclidean distance between two points: sqrt((x2-x1)^2 +(y2-y1)^2)
def Distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# Function to determine if the path between a ball and the robot crosses the obstacle
def CrossesObstacle(robot, ball, obstacles):
    # Extract coordinates of the robot and the ball
    rx, ry = robot
    bx, by = ball
    
    for obstacle in obstacles:
        (x1, y1), (x2, y2) = obstacle # Each obstacle is defined by two points (two points to create one of the cross lines)
        
        # Check if the x-coordinates of the robot and ball are on opposite sides of the obstacle
        if (rx < x1 and bx > x2) or (rx > x1 and bx < x2):
            # Check if the y-coordinates of the robot and ball are on opposite sides of the obstacle
            if (ry < y1 and by > y2) or (ry > y1 and by < y2):
               
                return True # If both conditions are true, the path crosses the obstacle
    return False # If no obstacles are crossed, return False


# Function to calculate the distance with a penalty for crossing the obstacle
def DistanceWithPenalty(robot, ball, obstacles):
    BaseDistance = Distance(robot, ball)
    penalty = 50  # Arbitrary penalty value for crossing the obstacle
    
    if CrossesObstacle(robot, ball, obstacles):
        return BaseDistance + penalty
    
    return BaseDistance


# Function to sort the positions of the balls based on their distance from the robot and applying a penalty for crossing the obstacle
def SortByDistance(RobotXY, BallsXY, OrangeBallXY, ObstacleXY):
    # Sort the balls based on distance to the robot with penalty for crossing the obstacle
    SortedList = sorted(BallsXY, key=lambda ball: DistanceWithPenalty(RobotXY, ball, ObstacleXY))
    
    # Add the orange ball at the end
    SortedList.append(OrangeBallXY)
    
    return SortedList



#******************************* FOR TESTING ******************************

# Print unsorted list
print("Unsorted list:", BallsXY)

# Sort the balls and print the sorted list
SortedExample = SortByDistance(RobotXY, BallsXY.copy(), OrangeBallXY, ObstacleXY)

print("Sorted list:", SortedExample)


# Function to move the robot to the target position
def move_to_target(target_position):
    print(f"Moving to target position: {target_position}")
    
# Call the function to move to the target
FirstBallPosition = SortedExample[0]
move_to_target(FirstBallPosition)

#******************************* FOR TESTING ******************************
