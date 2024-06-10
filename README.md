# Overview of the files and methods
The purpose of this section, is to give you a brief understanding of your life..

# Files

## home
1. ***run.py*** - The main of the program  
2. ***.gitignore*** - Blacklistes stuff we don't want to upload to github, like temporary files
3. ***.json*** - different local json files that are created when the program runs

## picture
1. ***image_detection.py*** all methods for detecting colors and shapes  
2. ***livefeed.py*** initialize camara and gets image to python
3. ***transform_arena.py*** used to make image correction, by transforming the perspective to some calibrated points.  

## extra
***test methods we might need later***


### image_detection.py
`def egg_draw(image, x, y, w, h, area):` - Draws the egg's circumference.

`def circle_detection(image):` - Detects a fixed size range of circles matching with the size of the ping-pong balls.

`def detect_ball_colors(image):` - The previous found color ranges are here used in order to differentiate between the specific colors in the the photo. Also the red cross is located with size and rotation, and the coordinates are saved to a json file

`def save_balls(circles, filename)` - Three functions to save balls, orange balls and white balls to seperate json files.

`def save_no_go_zones(zones, filename)` - Saves cross location to json

`def check_point_in_orange_region(contours)` - Check if balls are orange

###  run.py
`def transform_and_detect(image):` - Transforms image, and detects balls, cross and egg
`init()` - Initializes camara and manuallly find corners of arena

### livefeed.py
Someone write something plz
