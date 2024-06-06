Please don't read this


# Overview of ppArena
The purpose of this section, is to give you a brief understanding of your life..

## Files
1. ***run.py*** is the program to execute.  
2. ***utils.py*** every function is primarily stored in here.  
3. ***perspective_transform.py*** used to make image correction, by transforming the  perspective.  
4. ***get_corners.py*** manually choose the points in the image in order to do an image correction (only for test purposes).  
5. ***arena.py***  same as perspective_transform.py, here it is just incorporated with the rest of the code (or tried to :3).     




The images that are used is stored in test/images.

_________________________________

### utils.py
`def calibrateColors2(image)`: - Used for calibrating the given image and find the correct colors manually.


`def arena_draw(image, x, y, w, h, area):` - Draws the arenas circumference based on another function that detects the red color in the photo. Therefore the biggest red obstacle in photo, is the arenas circumference.

`def square_draw(image, x, y, w, h, area):` - Used for the red cross.

`def egg_draw(image, x, y, w, h, area):` - Draws the egg's circumference.

`def circle_detection(image):` - Detects a fixed size range of circles matching with the size of the ping-pong balls.

`def detect_ball_colors(image):` - The previous found color ranges are here used in order to differentiate between the specific colors in the the photo. 


###  run.py
`def basicDetectofImage():` - Detects the obstacles, all basic. I.e. the balls, egg, and colors.

`def getMeSomeBallInfo():` - Enable or disable the perspective transform. But this function is used to to image correction, to detect balls, egg, colors, and get the xy coordinates from the balls.

