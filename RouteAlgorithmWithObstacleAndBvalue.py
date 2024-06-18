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
        
    # # Extract the coordinates
    # obstacles = data[:-1]  # Ignore the last element (center with angle)
    # center_point = tuple(data[-1][0])  # Extract the center coordinates
    # cross_angle = data[-1][1]  # Extract the angle
    # return obstacles, center_point, cross_angle

# # Load the coordinates from the JSON files
# BallsXY = LoadCoordinatesWindows("whiteballs.json")
# OrangeBallXY = LoadCoordinatesWindows("orangeballs.json")[0]  # Only one orange ball, therefore take only the first element
# RobotXY = LoadCoordinatesWindows("robot.json")[0][:2]  # Only one robot, therefore take only the first element (without the angle)
# ObstacleXY, CenterXY = LoadObstaclesWindows(filename="no_go_zones.json")


# Converting the nested list that represents the points of the cross to a list
# ListObstacle = [point for sublist in ObstacleXY for point in sublist]
# # Debugging
# print("Converted ObstacleXY:", ListObstacle) 
# # Check the length of converted list for debugging 
# if len(ListObstacle) < 4:
#     raise ValueError("ERROR! Converted ObstacleXY does not contain enough points. Check the contents of no_go_zones.json.")


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
    obstacles = data[:-1]  # Ignore the last element (center with angle)
    center_point = tuple(data[-1][0])  # Extract the center coordinates
    cross_angle = data[-1][1]  # Extract the angle
    return obstacles, center_point, cross_angle


# Load the coordinates from the json files
BallsXY = LoadCoordinates("whiteballs.json")
OrangeBallXY = LoadCoordinates("orangeballs.json")[0]  # Only one orange ball, therefore take only the first element
RobotXY = LoadCoordinates("robot.json")[0][:2]  # Only one robot, therefore take only the first element (without the angle)
ObstacleXY, CenterPoint, CrossAngle = LoadObstacles("no_go_zones.json")
GoalsXY = LoadCoordinates("goals.json")

# Converting the nested list that represents the points of the cross to a list
ListObstacle = [point for sublist in ObstacleXY for point in sublist]
# # Debugging
# print("Converted ObstacleXY:", ListObstacle) 
# # Check the length of converted list for debugging 
# if len(ListObstacle) < 4:
#     raise ValueError("ERROR! Converted ObstacleXY does not contain enough points. Check the contents of no_go_zones.json.")

# Print the cross angle for debugging
print("Cross angle:", CrossAngle)

#********************************* FOR IBTI ******************************** 



# Defining the corners and edges of the track 
TrackCorners = {
    (0, 0): 1,      # Top left corner 
    (1250, 0): 2,   # Top right corner  
    (0, 900): 3,    # Bottom left corner 
    (1250, 900): 4  # Bottom right corner 
} 

TrackEdges = {
    ((0, 0), (1250, 0)): 5,      # Top edge
    ((0, 0), (0, 900)): 6,       # Left edge
    ((0, 900), (1250, 900)): 7,  # Bottom edge
    ((1250, 0), (1250, 900)): 8  # Right edge
}

# Defining the goal coordinates
Goal1 = tuple(GoalsXY[0])  # Center of edge 6
Goal2 = tuple(GoalsXY[1])  # Center of edge 8

# Print the goals for debugging
print("Goal 1:", Goal1)
print("Goal 2:", Goal2)

# Defining the points of the cross
ObstaclePoints = {
    tuple(ListObstacle[1]): 9,  # Top point 
    tuple(ListObstacle[0]): 10,  # Bottom point  
    tuple(ListObstacle[2]): 11,  # Left point 
    tuple(ListObstacle[3]): 12   # Right point
}


# # Calculating the center point of the cross(obstacle)
# def CalculateCenterPoint(ObstaclePoints):
#     TopPoint = tuple(ListObstacle[1])
#     BottomPoint = tuple(ListObstacle[0])
#     LeftPoint = tuple(ListObstacle[2])
#     RightPoint = tuple(ListObstacle[3])
    
#     CenterX = (LeftPoint[0] + RightPoint[0]) / 2
#     CenterY = (TopPoint[1] + BottomPoint[1]) / 2
#     CenterPoint = (CenterX, CenterY)
    
#     return CenterPoint

# CenterPoint = CalculateCenterPoint(ObstaclePoints)


# Function to calculate the distance between the Robot and the balls
# This function is based on the Euclidean distance between two points: sqrt((x2-x1)^2 +(y2-y1)^2)
def Distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# Function to determine the Bvalue of a ball based on its position
def BallValue(ball, TrackCorners, TrackEdges, ObstaclePoints, CenterPoint):
    # Check proximity to track corners
    for point, value in TrackCorners.items():
        if Distance(ball, point) < 100:  # Arbitrary distance to be considered "close"
            return value
    
    # Check proximity to track edges
    for ((x1, y1), (x2, y2)), value in TrackEdges.items():
        if x1 == x2:  # Vertical edge
            if abs(ball[0] - x1) < 100:  # Vertical edge proximity
                return value
        elif y1 == y2:  # Horizontal edge
            if abs(ball[1] - y1) < 100:  # Horizontal edge proximity
                return value
    
    # Determine which inner quadrant the ball is in
    TopPoint = tuple(ListObstacle[1])
    BottomPoint = tuple(ListObstacle[0])
    LeftPoint = tuple(ListObstacle[2])
    RightPoint = tuple(ListObstacle[3])

    # Check if the ball is close to lines forming the inner quadrants
    if Distance(ball, TopPoint) < 100 and Distance(ball, CenterPoint) < 100 and Distance(ball, LeftPoint) < 100:
        return 13
    
    # Check if the ball is in the top-right inner corner
    if Distance(ball, TopPoint) < 100 and Distance(ball, CenterPoint) < 100 and Distance(ball, RightPoint) < 100:
        return 14
    
    # Check if the ball is in the bottom-left inner corner
    if Distance(ball, LeftPoint) < 100 and Distance(ball, CenterPoint) < 100 and Distance(ball, BottomPoint) < 100:
        return 15
    
    # Check if the ball is in the bottom-right inner corner
    if Distance(ball, BottomPoint) < 100 and Distance(ball, CenterPoint) < 100 and Distance(ball, RightPoint) < 100:
        return 16
    
    # Check proximity to obstacle points (outer points of the cross)
    for point, value in ObstaclePoints.items():
        if Distance(ball, point) < 100:  # Arbitrary distance to be considered "close"
            return value

    # Default value for balls not close to any obstacle (track or cross)
    return 0


#******************************* FOR TESTING ******************************

Bvalue = [(ball, BallValue(ball, TrackCorners, TrackEdges, ObstaclePoints, CenterPoint)) for ball in BallsXY]
OrangeBvalue = (OrangeBallXY, BallValue(OrangeBallXY, TrackCorners, TrackEdges, ObstaclePoints, CenterPoint))

# Print the values for each ball
print("Ball values with coordinates:", Bvalue)
print("Orange ball value with coordinates:", OrangeBvalue)

# Plot the balls and their Bvalues
def PlotBallsAndValues(BallsWithValues, OrangeBallWithValue, RobotXY):
    plt.figure(figsize=(12, 9))

    # Plot the robot
    plt.plot(RobotXY[0], RobotXY[1], 'ro', label='Robot')

    # Plot the white balls
    for ball, value in BallsWithValues:
        plt.plot(ball[0], ball[1], 'go')  # Green circles for white balls
        plt.text(ball[0] + 10, ball[1] + 10, str(value), fontsize=12, color='green')

    # Plot the orange ball
    orange_ball, orange_value = OrangeBallWithValue
    plt.plot(orange_ball[0], orange_ball[1], 'o', color='orange', label='Orange Ball')  # Orange circle for orange ball
    plt.text(orange_ball[0] + 10, orange_ball[1] + 10, str(orange_value), fontsize=12, color='orange')  # Orange text for value

    # Plot the obstacle
    x1, y1 = zip(*[ListObstacle[0], ListObstacle[1]])
    x2, y2 = zip(*[ListObstacle[2], ListObstacle[3]])
    plt.plot(x1, y1, 'r-', linewidth=5, label='Obstacle Line 1')
    plt.plot(x2, y2, 'b-', linewidth=5, label='Obstacle Line 2')

    # Set the limits of the plot
    plt.xlim(0, 1300)
    plt.ylim(900, 0)  # Invert the y-axis

    # Adding labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Ball Positions and Their Bvalues')

    # Add a legend
    plt.legend(loc='upper right')

    # Display the plot
    plt.grid(True)
    plt.show()

# Plot the balls and their Bvalues
PlotBallsAndValues(Bvalue, OrangeBvalue, RobotXY)

#******************************* FOR TESTING ******************************



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
    penalty = 500  # Arbitrary penalty value for crossing the obstacle
    
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

# Sort balls by distance considering the penalty for crossing obstacles
SortedBalls = SortByDistance(RobotXY, BallsXY.copy(), OrangeBallXY, ObstacleXY)

# Create a final list with coordinates and Bvalues
FinalList = [(ball, BallValue(ball, TrackCorners, TrackEdges, ObstaclePoints, CenterPoint)) for ball in SortedBalls]

# Print the final sorted list with coordinates and Bvalues
print("Final sorted list with coordinates and Bvalues:", FinalList)

#******************************* FOR TESTING ******************************





