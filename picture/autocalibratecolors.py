import cv2
import numpy as np
import json

# Initialize global variables
clicked_points = []
hsv_values_white = []
hsv_values_orange = []
hsv_values_red = []
hsv_values_white_led = []
hsv_values_blue_led = []
frame = None
display_frame = None
current_color = 'white'  # Start with white as the default color


# Function to handle mouse clicks
def click_event(event, x, y, flags, param):
    global display_frame, current_color
    if event == cv2.EVENT_LBUTTONDOWN:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_value = hsv[y, x]
        clicked_points.append((x, y))
        
        # Save the HSV value based on the current selected color
        if current_color == 'white':
            hsv_values_white.append(hsv_value)
        elif current_color == 'orange':
            hsv_values_orange.append(hsv_value)
        elif current_color == 'red':
            hsv_values_red.append(hsv_value)
        elif current_color == 'white_led':
            hsv_values_white_led.append(hsv_value)
        elif current_color == 'blue_led':
            hsv_values_blue_led.append(hsv_value)

        
        display_frame = frame.copy()
        cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('image', display_frame)

# Function to calculate the min and max HSV values with buffer
def calculate_hsv_range(hsv_values, buffer=15):
    hsv_array = np.array(hsv_values)
    min_h = max(hsv_array[:, 0].min() - buffer, 0)
    max_h = min(hsv_array[:, 0].max() + buffer, 179)
    min_s = max(hsv_array[:, 1].min() - buffer, 0)
    max_s = min(hsv_array[:, 1].max() + buffer, 255)
    min_v = max(hsv_array[:, 2].min() - buffer, 0)
    max_v = min(hsv_array[:, 2].max() + buffer, 255)
    return np.array([min_h, min_s, min_v]), np.array([max_h, max_s, max_v])

# Function to save the HSV ranges to a JSON file
#def save_hsv_ranges(hsv_values_white, hsv_values_orange, hsv_values_red, hsv_values_white_led, hsv_values_blue_led):
   


    

# Function to load the HSV ranges from a JSON file
def load_hsv_ranges(file_path):
    with open(file_path, 'r') as file:
        hsv_ranges = json.load(file)
    return hsv_ranges

# Function to create an HSV mask using loaded HSV ranges
def create_hsv_mask_from_ranges(hsv_ranges, color, frame):
    min_hsv = np.array(hsv_ranges[color]['min'])
    max_hsv = np.array(hsv_ranges[color]['max'])
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, min_hsv, max_hsv)
    kernel = np.ones((5, 5), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask_closed

# Function to display the image and set up mouse callback
def select_colors_and_create_mask(image):
    global frame, display_frame, current_color
    #frame = cv2.imread(image)
    frame = image
    #display_frame = frame.copy()
    if frame is None:
        raise ValueError(f"Image not found at path: {image}")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)

    colors = ['white', 'orange', 'red', 'white_led', 'blue_led']
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

    white_min, white_max = calculate_hsv_range(hsv_values_white)
    mask = create_hsv_mask_from_ranges(white_min, white_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Debug: Draw all contours to visualize
    debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("contour_white.jpg", debug_image)
    orange_min, orange_max = calculate_hsv_range(hsv_values_orange)
    mask = create_hsv_mask_from_ranges(orange_min, orange_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Debug: Draw all contours to visualize
    debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("contour_orange.jpg", debug_image)
    red_min, red_max = calculate_hsv_range(hsv_values_red)
    mask = create_hsv_mask_from_ranges(red_min, red_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Debug: Draw all contours to visualize
    debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("contour_red.jpg", debug_image)
    white_led_min, white_led_max = calculate_hsv_range(hsv_values_white_led)
    mask = create_hsv_mask_from_ranges(white_led_min, white_led_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Debug: Draw all contours to visualize
    debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("contour_wled.jpg", debug_image)
    blue_led_min, blue_led_max = calculate_hsv_range(hsv_values_blue_led)
    mask = create_hsv_mask_from_ranges(blue_led_min, blue_led_max)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Debug: Draw all contours to visualize
    debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite("contour_bled.jpg", debug_image)

    hsv_ranges = {
        'red': (red_min,red_max),
        'white': (white_min,white_max),
        'orange': (orange_min,orange_max),
        'blue_LED': (blue_led_min,blue_led_max),
        'LED': (white_led_min,white_led_max)
    }
    np.savez('hsv_ranges.npz', **hsv_ranges)

image_path = "test1.jpg"
image = cv2.imread(image_path)

select_colors_and_create_mask(image)