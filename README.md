Please don't read this



# Overview of ppArena
This section is to give a brief understanding of the different functions in each file.

## Files

### utils.py
`def calibrateColors2(image)`: - Used for calibrating the given image and find the correct colors manually.


`def arena_draw(image, x, y, w, h, area):` - Draws the arenas circumference based on another function that detects the red color in the photo. Therefore the biggest red obstacle in photo, is the arenas circumference.

`def square_draw(image, x, y, w, h, area):` - Used for the red cross.

`def egg_draw(image, x, y, w, h, area):` - Draws the egg's circumference.

`def circle_detection(image):` - Detects a fixed size range of circles matching with the size of the ping-pong balls.

`def detect_ball_colors(image):` - The previous found color ranges are here used in order to differentiate between the specific colors in the the photo. 


## run.py
`def basicDetectofImage():` - Detects the obstacles, all basic. i.e. the balls, egg, and colors.

`def getMeSomeBallInfo():` - Enable or disable the perspective transform. But this function is used to to image correction, to detect balls, egg, colors, and get the xy coordinates from the balls.

