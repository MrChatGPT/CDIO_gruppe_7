# publish_controller_data(0.1, 0.1,   0.1,   0,    0)
#                          x,    y,   phi,  eat,  eject
#publish_controller_data(0.1,0.1,0.1,0,0)


import json
import sleep

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
    car_file_path = 'car.json'
    car = get_car_data_from_json(car_file_path)

    # Extract the current position from the car object
    current_x, current_y = car.x, car.y
    
    # De-structure the target position
    target_x, target_y = target_position
    print(f"Target_x = {target_x}\nTarget_y = {target_y}")


    # Move in the y direction
    if current_y != target_y:
        if current_y < target_y: #vi skal altså op ad ^
            #overvej at smid et threshold ind her
            if car.angle != 0:
                if car.angle > 0:
                    publish_controller_data(0,0,-0.15,0,0) #vi vender os mod venstre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
                if car.angle > 270:
                    publish_controller_data(0,0,0.15,0,0) #vi vender os mod højre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
            publish_controller_data(0,0.15,0,0,0) #vi rykker 0.15 i y-retningen (opad)
            sleep(0.5)
            publish_controller_data(0,0,0,0,0)
            return
        else: #så skal vi altså nedad ˅
            if car.angle != 180:
                if car.angle > 0:
                    publish_controller_data(0,0,0.15,0,0) #vi vender tilføjer grader indtil vi er på 180
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
                if car.angle > 180:
                    publish_controller_data(0,0,-0.15,0,0)#vi "fjerner" grader indtil vi er på 180
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
            publish_controller_data(0,0.15,0,0,0) #vi rykker 0.15 i y-retningen (opad)
            sleep(0.5)
            publish_controller_data(0,0,0,0,0)
            return
    
    # Move in the x direction
    # nu er vores angle enten 0 eller 180:
    if current_x != target_x:
        if current_x > target_x: #så skal vi til venstre
            if car.angle != 270:
                if car.angle == 0:
                    publish_controller_data(0,0,-0.15,0,0) #hvis vi peger opad ved 0 grader, skal vi bare tilte mod venstre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
                if car.angle > 270: #efter at have trukket én fra 0, lander vi på 359 grader
                    publish_controller_data(0,0,-0.15,0,0) #Hvis vi er større end 270, skal vi bare blive ved med at tilte mod venstre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
                if car.angle == 180:
                    publish_controller_data(0,0,0.15,0,0) #hvis vi allerede er ved 180, skal vi bare mod højre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return    
                if car.angle < 270:
                    publish_controller_data(0,0,0.15,0,0) #Hvis vi er større end 180, skal vi fortsat mod højre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
            publish_controller_data(0,0.15,0,0,0)
            sleep(0.5)
            publish_controller_data(0,0,0,0,0)
            return
        else: #så skal vi til højre
            if car.angle != 90:
                if car.angle == 180:
                    publish_controller_data(0,0,-0.15,0,0) #hvis vi peger med næsen nedad, skal vi tilte med bilens venstre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return    
                if car.angle > 90:
                    publish_controller_data(0,0,-0.15,0,0) #hvis vi peger med næsen nedad, skal vi tilte med bilens venstre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return    
                if car.angle == 0:
                    publish_controller_data(0,0,0.15,0,0) #hvis vi peger med næsen opad, skal vi tilte mod bilens højre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
                if car.angle < 90:
                    publish_controller_data(0,0,0.15,0,0) #hvis vi peger med næsen opad, skal vi tilte mod bilens højre
                    sleep(0.5)
                    publish_controller_data(0,0,0,0,0)
                    return
            publish_controller_data(0,0.15,0,0,0)
            sleep(0.5)
            publish_controller_data(0,0,0,0,0)
            return
    #nu er current_y = target_y, current_x = target_x
    publish_controller_data(0,0,0,1,0) #og så skal der nedsvælges
    sleep(0.5)
    publish_controller_data(0,0,0,0,0)


#Eksempel på kald af funktion: 
#move_to_target((2, -5))
