import cv2
import numpy as np
import json

# Initialize global variables
clicked_points = []
lab_values_white = []
lab_values_orange = []
lab_values_red = []
lab_values_green = []
frame = None
display_frame = None
current_color = 'white'  # Start with white as the default color

# Function to handle mouse clicks
def click_event(event, x, y, flags, param):
    global frame, display_frame, current_color
    if event == cv2.EVENT_LBUTTONDOWN:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_value = lab[y, x]
        clicked_points.append((x, y))
        
        # Save the LAB value based on the current selected color
        if current_color == 'white':
            lab_values_white.append(lab_value.tolist())
        elif current_color == 'orange':
            lab_values_orange.append(lab_value.tolist())
        elif current_color == 'red':
            lab_values_red.append(lab_value.tolist())
        elif current_color == 'green':
            lab_values_green.append(lab_value.tolist())
        
        display_frame = frame.copy()
        cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('image', display_frame)

# Function to calculate the min and max LAB values with buffer
def calculate_lab_range(lab_values, buffer=10):
    lab_array = np.array(lab_values)
    min_l = max(lab_array[:, 0].min() - buffer, 0)
    max_l = min(lab_array[:, 0].max() + buffer, 255)
    min_a = max(lab_array[:, 1].min() - buffer, 0)
    max_a = min(lab_array[:, 1].max() + buffer, 255)
    min_b = max(lab_array[:, 2].min() - buffer, 0)
    max_b = min(lab_array[:, 2].max() + buffer, 255)
    return np.array([min_l, min_a, min_b]), np.array([max_l, max_a, max_b])

# Function to save the LAB ranges to a JSON file
def save_lab_ranges():
    white_min, white_max = calculate_lab_range(lab_values_white)
    orange_min, orange_max = calculate_lab_range(lab_values_orange)
    red_min, red_max = calculate_lab_range(lab_values_red)
    green_min, green_max = calculate_lab_range(lab_values_green)

    lab_ranges = {
        'white': {'lower': white_min.tolist(), 'upper': white_max.tolist()},
        'orange': {'lower': orange_min.tolist(), 'upper': orange_max.tolist()},
        'red': {'lower': red_min.tolist(), 'upper': red_max.tolist()},
        'green': {'lower': green_min.tolist(), 'upper': green_max.tolist()}
    }
    with open('lab_ranges.json', 'w') as file:
        json.dump(lab_ranges, file)

# Function to load the LAB ranges from a JSON file
def load_lab_ranges(file_path):
    with open(file_path, 'r') as file:
        lab_ranges = json.load(file)
    return lab_ranges

# Function to create an LAB mask using loaded LAB ranges
def create_lab_mask_from_ranges(lab_ranges, color, frame):
    min_lab = np.array(lab_ranges[color]['lower'])
    max_lab = np.array(lab_ranges[color]['upper'])
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab_frame, min_lab, max_lab)
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask_closed

# Function to display the image and set up mouse callback
def select_colors_and_create_mask(image_path):
    global frame, display_frame, current_color
    frame = image_path
    display_frame = frame.copy()
    if frame is None:
        raise ValueError(f"Image not found at path: {image_path}")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)

    colors = ['white', 'orange', 'red', 'green']
    color_index = 0  # Start with the first color in the list

    while True:
        display_with_text = display_frame.copy()
        cv2.putText(display_with_text, f"Current Color: {current_color.capitalize()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', display_with_text)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            color_index = (color_index + 1) % len(colors)
            current_color = colors[color_index]

    cv2.destroyAllWindows()
    save_lab_ranges()

# Example usage:
# select_colors_and_create_mask('path_to_your_image.png')
