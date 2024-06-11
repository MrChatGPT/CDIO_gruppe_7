# Overview of the files and methods
The purpose of this section, is to give you a brief understanding of your life..

# Files

## home directory
1. ***run.py*** - The main of the program  
2. ***.gitignore*** - Blacklistes stuff we don't want to upload to github, like temporary files
3. ***.json*** - different local json files that are created when the program runs

## picture directory
1. ***image_detection.py*** all methods for detecting colors and shapes  
2. ***livefeed.py*** initialize camara and gets image to python
3. ***transform_arena.py*** used to make image correction, by transforming the perspective to some calibrated points.
4. ***algorithm.py*** used to create an algorithm and send car on it's way.

## extra directory
***test methods we might need later***


### image_detection.py
`def egg_draw(image, x, y, w, h, area):` - Draws the egg's circumference.

`def circle_detection(image):` - Detects a fixed size range of circles matching with the size of the ping-pong balls.

`def detect_ball_colors(image):` - The previous found color ranges are here used in order to differentiate between the specific colors in the the photo. Also the red cross is located with size and rotation, and the coordinates are saved to a json file

`def save_balls(circles, filename)` - Three functions to save balls, orange balls and white balls to seperate json files.

`def save_no_go_zones(zones, filename)` - Saves cross location to json

`def check_point_in_orange_region(contours)` - Check if balls are orange

`def find_car(image,output_path*,yellow_mask_path*,green_mask_path*,center_weight=25*):` - Returns a touple and creates a .json file with the cars center and angle. Everything marked with '*' is optional.

`def rgb_to_hsv(rgb):` - Gets RGB colors as a numpy.array as input, and returns the hsv color, good for creating masks.


###  run.py
`def transform_and_detect(image):` - Transforms image, and detects balls, cross and egg
`init()` - Initializes camara and manuallly find corners of arena

### livefeed.py
`def __init__(self):` - used as a "state" holder.

`def start_video(self):` - Used to initialize the camera, and spawn a thread.

`def capture_image(self):` - dead function, but can take a picture of the spawned window and save it to the current working directory.

`def release_camera(self):` - Used to kill thread, and stop the camera.

### algorithm.py (Patric_algo\Patric_funk\ibtiogpatricfunc.py)
`def get_car_data_from_json(file_path):` - Opens the robot.json file and puts the information inside an object.

`def move_to_target(target_position):` - Logic for moving the car giving a set of coordinates (x,y)

`def LoadBalls(filename="balls.json"):` - Opens the balls.json file and puts the information inside a touple.

`def LoadOrangeBall(filename="orangeball.json"):` - Opens the orangeball.json file and puts the information inside a touple (lonely one)

`def LoadRobot(filename="robot.json"):` - Opens the robot.json file and puts the coordinates inside a touple. (discards angle/alpha, since we don't need it here) 

`def Distance(p1, p2):` - The Pythagorean theorem.

`def SortByDistance(RobotXY, BallsXY):` - sort closest to furthest 0...9 and append orange ball after sort. (this is the first algorithm)



