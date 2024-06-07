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

`def goal_draw(image, x, y):` - Draws the goals based on the xy coordinates of the big red arena.

`def square_draw(image, x, y, w, h, area):` - Used for the red cross.

`def egg_draw(image, x, y, w, h, area):` - Draws the egg's circumference.

`def circle_detection(image):` - Detects a fixed size range of circles matching with the size of the ping-pong balls.

`def detect_ball_colors(image):` - The previous found color ranges are here used in order to differentiate between the specific colors in the the photo. 



###  run.py
`def basicDetectofImage():` - Detects the obstacles, all basic. I.e. the balls, egg, and colors.

`def getMeSomeBallInfo():` -  Detects the obstacles, all basic. I.e. the balls, egg, and colors and get the xy coordinates from the balls.

`def wBabyCanny(image):` - Used when transforming the perspective. First the arena is detected, and then detect the obstacles, all basic. I.e. the balls, egg, and colors.


### arena.py
`def detect_arena(image):` - Detects only the big red square (arena) in the picture.

`def perspectiveTransDyn(image,x,y,w,h):` - Used to determine the transformed picture, dynamically.