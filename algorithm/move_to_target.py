import json
from time import sleep
import numpy as np 
import os
from algorithm.control import publish_controller_data

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
    # Extract the current position from the car object
    current_x, current_y = car.x, car.y
    # add the x,y such that we go out to the pickup mechanism
   

    # De-structure the target position
    target_x, target_y = target_position
    #print(f"Target_x = {target_x}\nTarget_y = {target_y}")
    print(f"Desired position: {target_position}\nMy position: {current_x,current_y}, angle: {car.angle}")
    #Commands
    comstop = (0,0,0,0,0)
    comtiltleft = (0,0,-0.15,0,0)
    comtiltright = (0,0,0.15,0,0)
    comforward = (0,0.15,0,0,0)
    comswallow = (0,0,0,1,0)

    # Move in the y direction
    threshhold = 30
    missing_length = 0

    if not(((((current_y+missing_length)>(target_y-threshhold)) and ((current_y+missing_length)<(target_y+threshhold))) or (((current_y-missing_length)>(target_y-threshhold)) and ((current_y-missing_length)<(target_y+threshhold))))):
        print("missing in y-level")
        if current_y < target_y: #Hvis vores y er skarpt mindre end hvor vi skal hen-20 (hvilket burde resultere i at vi stopper cirka 20 pixels før.)
        #vi skal altså op ad /\ (robot y er mindre end target, altså er robot tættere på 0,0)
            if  not(((car.angle < (0+threshhold)) or (car.angle > (360-threshhold)))): #angle ligger ikke(mellem 340 og 20 grader (cirka 0)) 
                print("desired angle: 340-20, actual:", car.angle)
                publish_controller_data(comtiltright) #vi vender os mod højre
                sleep(0.2)
                publish_controller_data(comstop)
                return 0
            print("We should move farwards")
            publish_controller_data(comforward) #vi rykker 0.15 i y-retningen (opad)
            sleep(0.2)
            publish_controller_data(comstop)
            return 0
        elif current_y > target_y: #Hvis vores y er skarpt større end hvor vi skal hen+20 (hvilket burde resultere i at vi stopper cirka 20 pixels før.)
        #vi skal altså ned ad \/ (robot y er større end target, altså er bold tættere på 0,0)
            if not(((car.angle > (180-threshhold)) and (car.angle < (180+threshhold)))): #angle ligger ikke(mellem 170 og 190 grader(cirka 180))
                print("desired angle: 170-190, actual:", car.angle)
                publish_controller_data(comtiltright) #vi vender tilføjer grader indtil vi er på 180
                sleep(0.2)
                publish_controller_data(comstop)
                return 0
            print("We should move farwards 180")
            
            publish_controller_data(comforward) #vi rykker 0.15 i y-retningen (opad)
            sleep(0.2)
            publish_controller_data(comstop)
            return 0
        print("overshooted in y-level")
        return 0
    
    # Move in the x direction
    # nu er vores angle cirka 0 eller cirka 180: (340-20 eller 160-200)

    #if not((current_x>target_x+missing_length) or (current_x<target_x)):    
    if not(((((current_x-missing_length)>(target_x+threshhold)) and ((current_x-missing_length)<(target_x-threshhold))) or (((current_x+missing_length)>(target_x-threshhold)) and ((current_x+missing_length)<(target_x+threshhold))))):
        print("missing in x-level")
        if current_x > target_x: #så skal vi til venstre
            if not(((car.angle < (90+threshhold)) and (car.angle > (90-threshhold)))):
                publish_controller_data(comtiltleft) #hvis vi peger opad ved 0 grader, skal vi bare tilte mod venstre
                sleep(0.2)
                publish_controller_data(comstop)
                return 0
            publish_controller_data(comforward)
            sleep(0.2)
            publish_controller_data(comstop)
            return 0
        #så skal vi til højre
        elif current_x < target_x:
            if not((car.angle < (270+threshhold)) and (car.angle > (270-threshhold))): #hvis car.angle ikke(er mellem 250 og 290 (cirka 270))
                publish_controller_data(comtiltright) #hvis vi peger med næsen opad, skal vi tilte mod bilens højre
                sleep(0.2)
                publish_controller_data(comstop)
                return 0
            publish_controller_data(comforward)
            sleep(0.2)
            publish_controller_data(comstop)
            return 0
        #nu er current_y = target_y, current_x = target_x
        print("overshooted in x-level...")
        return 0
        
    print("we're here!")
    publish_controller_data(comswallow) #og så skal der nedsvælges
    sleep(0.2)
    publish_controller_data(comstop)
    return 1




