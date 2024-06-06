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
objects = [Cross(), Car(0, 0, 0), Egg()]

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
        if current_y < target_y:
            #adjust_angle(car, 0)
            if car.angle == 0:
                adjust_angle(car, 0)
            elif car.angle == 90:
                adjust_angle(car, -90)
            elif car.angle == 180:
                adjust_angle(car, 180)
            elif car.angle == 270:
                adjust_angle(car, 90)
            while current_y < target_y:
                move_forward(car, 1)
                current_y += 1
        else:
            adjust_angle(car, 180)
            while current_y > target_y:
                move_forward(car, 1)
                current_y -= 1
    
    # Move in the x direction
    if current_x != target_x:
        if current_x < target_x:
            #adjust_angle(car, 90)
            if car.angle == 0:
                adjust_angle(car, 90)
            elif car.angle == 90:
                adjust_angle(car, 0)
            elif car.angle == 180:
                adjust_angle(car, -90)
            elif car.angle == 270:
                adjust_angle(car, 180)
            while current_x < target_x:
                move_forward(car, 1)
                current_x += 1
        else:
            if car.angle == 0:
                adjust_angle(car, -90)
            elif car.angle == 90:
                adjust_angle(car, 180)
            elif car.angle == 180:
                adjust_angle(car, 90)
            elif car.angle == 270:
                adjust_angle(car, 0)
            while current_x > target_x:
                move_forward(car, 1)
                current_x -= 1
    
    # Update the car's position
    car.x, car.y = current_x, current_y
    print(f"New position of the car: ({car.x}, {car.y}) with angle {car.angle}")

def move_forward(car, step):
    print(f"Moving forward by {step} step(s) at angle {car.angle}")
    # Logic to move forward considering current angle
    if car.angle == 0:
        car.y += step
    elif car.angle == 180:
        car.y -= step
    elif car.angle == 90:
        car.x += step
    elif car.angle == 270:
        car.x -= step

def adjust_angle(car, new_angle):
    if car.angle+new_angle < 0:
        new_angle = 270
    print(f"Adjusting angle from {car.angle} to {car.angle+new_angle} degrees")
    car.angle += new_angle % 360
    if car.angle == 360:
        car.angle = 0


# Example usage
#my pos = 3,5   targ pos = 0,0 expected angle = 180+90 = 270
target_position = (0, 0)
move_to_target(target_position)
#my pos = 0,0 targ pos = 3,5 expected angle = 0+90 = 90
target_position = (3, 5)
move_to_target(target_position)
#my pos = 3,5 targ pos = 2,5 expected angle = 0+270 = 270 
target_position = (2, 5)
move_to_target(target_position)
#my pos = 2,5 targ pos = 4,6 expected angle = 0+90 = 90
target_position = (4, 6)
move_to_target(target_position)