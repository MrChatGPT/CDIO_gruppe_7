import cv2
import numpy as np
import json
import time

def save_calibration_data(file_path, data):
    # mrChat shit to work with json
    data_list = [point.tolist() for point in data]
    with open(file_path, 'w') as file:
        json.dump(data_list, file)

def load_calibration_data(file_path):
    with open(file_path, 'r') as file:
        data_list = json.load(file)
    # Convert list of lists back to numpy array
    return np.array(data_list, dtype=np.float32)

# mouse click event handler
def draw_circle(event, x, y, flags, param):
    img, clicked_points = param # img was passed in param
    if event == cv2.EVENT_LBUTTONDOWN:
        # Save points
        clicked_points.append((x, y))
        # Draw circle 
        cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

def find_corners(img):
    # Initialize a list to store the coordinates of the corners
    clicked_points = [] 
    # Set up event handler
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_circle, (img, clicked_points))

    while True:
        # Display the image
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        # Break the loop when 'c' is pressed or 4 points are put
        if len(clicked_points) == 4 or key == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(clicked_points) != 4:
        print("You fucked up son")
        return

    points  = np.float32(clicked_points)

    # crazy over-engeneerd sort of points
    centroid = np.mean(points, axis=0)
    # Categorize points relative to the centroid
    def categorize_point(point):
        if point[0] < centroid[0] and point[1] < centroid[1]:
            return "top_left"
        elif point[0] > centroid[0] and point[1] < centroid[1]:
            return "top_right"
        elif point[0] > centroid[0] and point[1] > centroid[1]:
            return "bottom_right"
        else:
            return "bottom_left"
    categorized_points = {categorize_point(point): point for point in points}
    sorted_corners = [
        categorized_points["top_left"],
        categorized_points["top_right"],
        categorized_points["bottom_right"],
        categorized_points["bottom_left"]
    ]
    sorted_corners = np.array(sorted_corners, dtype="float32")

    #save for later use
    save_calibration_data('calibration_data.json', sorted_corners)
    transform(img)

def transform(img):
    # resulution (can be made smaller if we want to)
    # the arena is 180 * 120, so we multiply by 5 for this
    width, height = 1395, 975

    corners = load_calibration_data('calibration_data.json')
    input_corners = np.float32(corners)

    correct_corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    # Get the transformation matrix
    matrix = cv2.getPerspectiveTransform(input_corners, correct_corners)

    # transform
    transformed = cv2.warpPerspective(img, matrix, (width, height))

    cv2.imshow('Transformed Image', transformed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return transformed

def calibrate(img):
        backup = img.copy()
        find_corners(img)
        # while True:
        #     calibration = input("Would you like to redo the calibration?[y/n]:")
        #     if calibration == "y":
        #         img = backup.copy()
        #         find_corners(img)
        #     else:
        #         break

# calibrate(cv2.imread('ppArena/test/images/WIN_20240410_10_31_07_Pro.jpg'))
# transform(cv2.imread('ppArena/test/images/WIN_20240410_10_31_07_Pro.jpg'))
