import cv2
import numpy as np

image = cv2.imread('test/images/WIN_20240410_10_31_07_Pro.jpg') 

# input coordinates, they are just hard coded for now (genereted in get_corners.py) :3
#top-left, top-right, bottom-right, bottom-left
input_corners = np.float32([
    [372, 15],
    [1751, 17],
    [1751, 1045],
    [345, 1016]  
])

# Width and height of the new image
width = 1600
height = 1000
correct_corners = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])

# Get the transformation matrix
matrix = cv2.getPerspectiveTransform(input_corners, correct_corners)

# transform
transformed = cv2.warpPerspective(image, matrix, (width, height))

cv2.imshow('Transformed Image', transformed)
cv2.waitKey(0)
cv2.destroyAllWindows()
