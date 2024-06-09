# This code was inspired by:
# https://www.varsitytutors.com/hotmath/hotmath_help/topics/distance-formula
# https://www.youtube.com/watch?v=CWUr6Jo6tag 
# https://docs.python.org/3/howto/sorting.html




import json
import numpy as np 


# # Example for Robot position 
# # This value should be updated in coordinance with picture-update

# RobotXY = (0, 0)


# # The list containing the white balls' position 

# BallsXY = [
#     (2, 3),     # Ball 1
#     (5, 1),     # Ball 2
#     (1, 4),     # Ball 3
#     (4, 6),     # Ball 4
#     (6, 2),     # Ball 5
#     (3, 7),     # Ball 6
#     (8, 3),     # Ball 7
#     (7, 5),     # Ball 8
#     (9, 1),     # Ball 9
# ]

# # Position of the orange ball
# OrangeBallXY = (7, 8)  # Ball 10 (orange ball)



# Function to read ball positions from a JSON file
def LoadPositions(filename="balls.json"):
    with open(filename, 'r') as file:
        data = json.load(file)
    # Convert lists to tuples
    RobotXY = tuple(data["RobotXY"])
    BallsXY = [tuple(center) for center in data["BallsXY"]]
    OrangeBallXY = tuple(data["OrangeBallXY"])
    return RobotXY, BallsXY, OrangeBallXY


# Loading the positions from the JSON file
RobotXY, BallsXY, OrangeBallXY = LoadPositions()


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
ProcessFirstBall(FirstBallPosition)














