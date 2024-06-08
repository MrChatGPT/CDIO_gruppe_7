# publish_controller_data(0.1, 0.1,   0.1,   0,    0)
#                          x,    y,   phi,  eat,  eject
#publish_controller_data(0.1,0.1,0.1,0,0)

# Example classes for objects
class Car:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle

class Cross:
    pass

class Egg:
    pass

# Example list of objects including a cross, a car, and an egg
objects = [Cross(), Car(3, 5, 0), Egg()]

def move_to_target(target_position):
    # Find the car object in the list
    car = next((obj for obj in objects if isinstance(obj, Car)), None)
    
    if not car:
        print("Car not found in the list")
        return
    
    # Extract the current position from the car object
    current_x, current_y = car.x, car.y
    
    # Destructure the target position
    target_x, target_y = target_position
    
    # Move in the y direction
    if current_y != target_y:
        if current_y < target_y: #vi skal altså op ad ^
            #overvej at smid et threshold ind her
            if car.angle > 0:
                publish_controller_data(0,0,-0.1,0,0) #vi vender os mod venstre
                return
            if car.angle > 270:
                publish_controller_data(0,0,0.1,0,0) #vi vender os mod højre
                return
            publish_controller_data(0,0.1,0,0,0) #vi rykker 0.1 i y-retningen (opad)
            return
        else: #så skal vi altså nedad ˅
            if car.angle != 180:
                if car.angle > 0:
                    publish_controller_data(0,0,0.1,0,0) #vi vender tilføjer grader indtil vi er på 180
                    return
                if car.angle > 180:
                    publish_controller_data(0,0,-0.1,0,0)#vi "fjerner" grader indtil vi er på 180
                    return
        publish_controller_data(0,0.1,0,0,0) #vi rykker 0.1 i y-retningen (opad)
        return
    
    # Move in the x direction
    # nu er vores angle enten 0 eller 180:
    if current_x != target_x:
        if current_x > target_x: #så skal vi til venstre
            if car.angle != 270:
                if car.angle == 0:
                    publish_controller_data(0,0,-0.1,0,0) #hvis vi peger opad ved 0 grader, skal vi bare tilte mod venstre
                    return
                if car.angle > 270: #efter at have trukket én fra 0, lander vi på 359 grader
                    publish_controller_data(0,0,-0.1,0,0) #Hvis vi er større end 270, skal vi bare blive ved med at tilte mod venstre
                    return
                if car.angle == 180:
                    publish_controller_data(0,0,0.1,0,0) #hvis vi allerede er ved 180, skal vi bare mod højre
                    return    
                if car.angle > 180:
                    publish_controller_data(0,0,0.1,0,0) #Hvis vi er større end 180, skal vi fortsat mod højre
                    return
            publish_controller_data(0,0.1,0,0,0)
            return
        else: #så skal vi til højre
            if car.angle != 90:
                if car.angle == 180:
                    publish_controller_data(0,0,-0.1,0,0)
                    return    
                if car.angle > 90:
                    publish_controller_data(0,0,-0.1,0,0)
                    return    
                if car.angle == 0:
                    publish_controller_data(0,0,0.1,0,0) #hvis vi peger opad ved 0 grader, skal vi bare tilte mod højre
                    return
                if car.angle > 0:
                    publish_controller_data(0,0,0.1,0,0) #hvis vi er større end 0 skal vi også pege mod højre
                    return
            publish_controller_data(0,0.1,0,0,0)
            return
    #nu er current_y = target_y, current_x = target_x
    publish_controller_data(0,0,0,1,0) #og så skal der nedsvælges

