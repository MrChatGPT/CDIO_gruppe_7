import cv2
import numpy as np

# Initialize global variables
clicked_points = []
hsv_values_white = []
hsv_values_orange = []
hsv_values_red = []
hsv_values_green = []
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
        elif current_color == 'green':
            hsv_values_green.append(hsv_value)
        
        display_frame = frame.copy()
        cv2.circle(display_frame, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow('image', display_frame)
        print(f"Clicked at: ({x}, {y}) with HSV value: {hsv_value}")

# Function to create an HSV mask
def create_hsv_mask(hsv_values, frame, buffer=10):
    if not hsv_values:
        raise ValueError("No colors were selected. Please click on at least one color.")
    # Convert the list of HSV values to a NumPy array
    hsv_array = np.array(hsv_values)
    
    # Apply buffer separately to H, S, and V channels
    min_h = max(hsv_array[:, 0].min() - buffer, 0)
    max_h = min(hsv_array[:, 0].max() + buffer, 179)
    min_s = max(hsv_array[:, 1].min() - buffer, 0)
    max_s = min(hsv_array[:, 1].max() + buffer, 255)
    min_v = max(hsv_array[:, 2].min() - buffer, 0)
    max_v = min(hsv_array[:, 2].max() + buffer, 255)
    
    min_hsv = np.array([min_h, min_s, min_v])
    max_hsv = np.array([max_h, max_s, max_v])
    
    print(f"Min HSV: {min_hsv}, Max HSV: {max_hsv}")  # Print HSV ranges for verification
    
    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create the mask
    mask = cv2.inRange(hsv_frame, min_hsv, max_hsv)
    return mask


# Function to display the image and set up mouse callback
def select_colors_and_create_mask(image_path):
    global frame, display_frame, current_color
    frame = cv2.imread(image_path)
    display_frame = frame.copy()
    if frame is None:
        raise ValueError(f"Image not found at path: {image_path}")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', click_event)

    while True:
        # Update the display_frame with the current color text
        display_with_text = display_frame.copy()
        cv2.putText(display_with_text, f"Current Color: {current_color.capitalize()}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', display_with_text)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('w'):
            current_color = 'white'
        elif key == ord('o'):
            current_color = 'orange'
        elif key == ord('r'):
            current_color = 'red'
        elif key == ord('g'):
            current_color = 'green'

    cv2.destroyAllWindows()
    white_mask = create_hsv_mask(hsv_values_white, frame)
    orange_mask = create_hsv_mask(hsv_values_orange, frame)
    red_mask = create_hsv_mask(hsv_values_red, frame)
    green_mask = create_hsv_mask(hsv_values_green, frame)
    return white_mask, orange_mask, red_mask, green_mask

# Example usage:
try:
    white_mask, orange_mask, red_mask, green_mask = select_colors_and_create_mask('extra/test/images/WIN_20240618_11_28_17_Pro.jpg')
    print(f"White mask unique values: {np.unique(white_mask)}")
    print(f"Orange mask unique values: {np.unique(orange_mask)}")
    print(f"Red mask unique values: {np.unique(red_mask)}")
    print(f"Green mask unique values: {np.unique(green_mask)}")
    
    # Apply masks to the original image for visualization
    white_masked_image = cv2.bitwise_and(frame, frame, mask=white_mask)
    orange_masked_image = cv2.bitwise_and(frame, frame, mask=orange_mask)
    red_masked_image = cv2.bitwise_and(frame, frame, mask=red_mask)
    green_masked_image = cv2.bitwise_and(frame, frame, mask=green_mask)
    
    cv2.imshow('White Mask', white_mask)
    cv2.imshow('Orange Mask', orange_mask)
    cv2.imshow('Red Mask', red_mask)
    cv2.imshow('Green Mask', green_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except ValueError as e:
    print(e)