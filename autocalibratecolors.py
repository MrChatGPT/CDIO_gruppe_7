import cv2
import numpy as np
import json

# Initialize global variables
clicked_points = []
lab_values_white = []
lab_values_orange = []
lab_values_red = []
lab_values_green = []
lab_values_led = []  # Added for LED calibration
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
        elif current_color == 'led':  # Added for LED calibration
            lab_values_led.append(lab_value.tolist())
        
        display_frame = frame.copy()
        cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('image', display_frame)

# Function to calculate the average LAB values for lower and upper bounds
def calculate_avg_lab_range(lab_values):
    lab_array = np.array(lab_values)
    avg_l = int(np.mean(lab_array[:, 0]))
    avg_a = int(np.mean(lab_array[:, 1]))
    avg_b = int(np.mean(lab_array[:, 2]))
    return np.array([avg_l, avg_a, avg_b]), np.array([avg_l, avg_a, avg_b])

# Function to save the average LAB ranges to a JSON file
def save_avg_lab_ranges():
    white_avg = calculate_avg_lab_range(lab_values_white)
    orange_avg = calculate_avg_lab_range(lab_values_orange)
    red_avg = calculate_avg_lab_range(lab_values_red)
    green_avg = calculate_avg_lab_range(lab_values_green)
    led_avg = calculate_avg_lab_range(lab_values_led)  # Added for LED calibration

    lab_ranges = {
        'white': {'lower': white_avg[0].tolist(), 'upper': white_avg[1].tolist()},
        'orange': {'lower': orange_avg[0].tolist(), 'upper': orange_avg[1].tolist()},
        'red': {'lower': red_avg[0].tolist(), 'upper': red_avg[1].tolist()},
        'green': {'lower': green_avg[0].tolist(), 'upper': green_avg[1].tolist()},
        'led': {'lower': led_avg[0].tolist(), 'upper': led_avg[1].tolist()}  # Added for LED calibration
    }
    with open('avg_lab_ranges.json', 'w') as file:
        json.dump(lab_ranges, file)

# Function to load the average LAB ranges from a JSON file
def load_avg_lab_ranges(file_path):
    with open(file_path, 'r') as file:
        lab_ranges = json.load(file)
    return lab_ranges

# Function to create an LAB mask using loaded average LAB ranges
def create_lab_mask_from_avg_ranges(lab_ranges, color, frame):
    avg_lab = np.array(lab_ranges[color]['lower'])  # Both lower and upper are the same
    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab_frame, avg_lab, avg_lab)
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask_closed

# Function to display the image and set up mouse callback
def select_colors_and_create_mask(image_path):
    global frame, display_frame, current_color
    frame = cv2.imread(image_path)
    display_frame = frame.copy()
    if frame is None:
        raise ValueError(f"Image not found at path: {image_path}")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)

    colors = ['white', 'orange', 'red', 'green', 'led']  # Added 'led' for LED calibration
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
    save_avg_lab_ranges()

# Example usage:
# select_colors_and_create_mask('path_to_your_image.png')
