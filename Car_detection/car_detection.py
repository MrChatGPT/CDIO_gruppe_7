import cv2
import numpy as np
import math
import json
import os

def rgb_to_hsv(rgb):
    color = np.uint8([[rgb]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]


def detect_green_hsv_bounds(image_path):
    # Read the image
    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path))
    if image is None:
        raise ValueError("Image not found or unable to read the image file.")

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define a range for green color in HSV space
    # These are broad ranges and may need to be adjusted based on the image
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])

    # Create a mask for green color
    mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables to store the overall lower and upper bounds
    overall_lower_bound = np.array([255, 255, 255])
    overall_upper_bound = np.array([0, 0, 0])

    # Iterate over each contour to find the bounds
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = hsv_image[y:y+h, x:x+w]

        min_hsv = roi.min(axis=(0, 1))
        max_hsv = roi.max(axis=(0, 1))

        overall_lower_bound = np.minimum(overall_lower_bound, min_hsv)
        overall_upper_bound = np.maximum(overall_upper_bound, max_hsv)

    print("Lower HSV Bound for green detected: ", overall_lower_bound)
    print("Upper HSV Bound for green detected: ", overall_upper_bound)

    append_to_json('lower_bounds.json', overall_lower_bound.tolist())
    append_to_json('upper_bounds.json', overall_upper_bound.tolist())

    return overall_lower_bound, overall_upper_bound

def append_to_json(filename, data):
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            file_data = json.load(file)
    else:
        file_data = []

    file_data.append(data)

    with open(filename, 'w') as file:
        json.dump(file_data, file, indent=4)

def calculate_mean_from_json(filename):
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist.")

    with open(filename, 'r') as file:
        data = json.load(file)
    
    if not data:
        raise ValueError(f"No data found in {filename}.")

    np_data = np.array(data)
    mean_values = np.mean(np_data, axis=0)

    return mean_values

# Example usage
# Calculate mean values from JSON files
#mean_lower_bound = calculate_mean_from_json('lower_bounds.json')
#mean_upper_bound = calculate_mean_from_json('upper_bounds.json')

#print(f"Mean lower HSV bound: {mean_lower_bound}")
#print(f"Mean upper HSV bound: {mean_upper_bound}")



def find_car(image_path, output_path='output_image.jpg', yellow_mask_path='yellow_mask.jpg', green_mask_path='green_mask.jpg', center_weight=25):
    # Read the image
    
    image = cv2.imread(os.path.join(os.path.dirname(__file__), image_path))

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    



    # Define broader HSV ranges for green, gotten after using the two green_hsv1 and green_hsv2 
    green_lower_hsv = np.array([75, 100, 100])
    green_upper_hsv = np.array([95, 255, 255])

    # Create masks for yellow and green
    yellow_mask = cv2.inRange(hsv, yellow_lower_hsv, yellow_upper_hsv)
    green_mask = cv2.inRange(hsv, green_lower_hsv, green_upper_hsv)
    
    # Save the masks for debugging
    #cv2.imwrite(yellow_mask_path, yellow_mask)
    #cv2.imwrite(green_mask_path, green_mask)
    
    # Find contours for yellow and green regions
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours_green, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Function to find the centroid of the largest contour
    def find_centroid(contours):
        if len(contours) == 0:
            return None
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        return (cx, cy)
    
    # Find centroids of the largest yellow and green contours
    yellow_centroid = find_centroid(contours_yellow)
    green_centroid = find_centroid(contours_green)
    
    if yellow_centroid is None or green_centroid is None:
        raise ValueError("Could not find the required yellow or green regions in the image.")
    
    # Calculate the center of the car
    center_x = (yellow_centroid[0] + green_centroid[0]) // 2
    center_y = (yellow_centroid[1] + green_centroid[1]) // 2
    
    # Adjust the center position based on the center_weight
    line_vec_x = green_centroid[0] - yellow_centroid[0]
    line_vec_y = green_centroid[1] - yellow_centroid[1]
    line_length = np.sqrt(line_vec_x ** 2 + line_vec_y ** 2)
    
    unit_vec_x = line_vec_x / line_length
    unit_vec_y = line_vec_y / line_length
    
    adjusted_center_x = center_x + int(center_weight * unit_vec_x)
    adjusted_center_y = center_y + int(center_weight * unit_vec_y)
    
    # Calculate the angle of the car with respect to (0,0)
    angle_rad = math.atan2(-line_vec_y, line_vec_x)  # Invert y to account for image coordinate system
    angle_deg = math.degrees(angle_rad)+90
   
    # Ensure the angle is in the range [0, 360)
    if angle_deg < 0:
        angle_deg += 360
    
    #if we wish the angle to be a rounded integer (ex: 180.7010 = 181, 180.46 = 180):
    angle_deg = int(round(angle_deg))
    
    # Draw the centroids, car center, and direction line on the image for visualization
    cv2.circle(image, yellow_centroid, 5, (0, 255, 255), -1) # Yellow centroid
    cv2.circle(image, green_centroid, 5, (0, 255, 0), -1)   # Green centroid
    cv2.circle(image, (adjusted_center_x, adjusted_center_y), 5, (255, 0, 0), -1) # Car center
    cv2.line(image, green_centroid, yellow_centroid, (255, 0, 0), 2) # Direction line
    
    # Save the result
    cv2.imwrite(os.path.join(os.path.dirname(__file__), output_path), image)
    
    # Write the results to a JSON file
    data = [[adjusted_center_x, adjusted_center_y, angle_deg]]
    with open('robot.json', 'w') as json_file:
        json.dump(data, json_file)

    return (adjusted_center_x, adjusted_center_y, angle_deg)


# Example usage:
# Example usage:
#image_path = 'image_one.jpg'
#image_path = 'image_two.jpg'
#image_path = 'image_three.jpg'
#image_path = 'image_four.jpg'
#image_path = 'image_five.jpg'
#image_path = 'image_six.jpg'
#image_path = 'image_seven.jpg'
#image_path = 'image_eight.jpg'
#image_path = 'image_nine.jpg'
# Example usage
#image_path = 'path_to_your_image.jpg'


def calculate_median_from_json(filename):
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist.")

    with open(filename, 'r') as file:
        data = json.load(file)
    
    if not data:
        raise ValueError(f"No data found in {filename}.")

    np_data = np.array(data)
    median_values = np.median(np_data, axis=0)

    return median_values


def find_max_from_json(filename):
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist.")

    with open(filename, 'r') as file:
        data = json.load(file)
    
    if not data:
        raise ValueError(f"No data found in {filename}.")

    np_data = np.array(data)
    max_values = np.max(np_data, axis=0)

    return max_values

def find_min_from_json(filename):
    if not os.path.exists(filename):
        raise ValueError(f"{filename} does not exist.")

    with open(filename, 'r') as file:
        data = json.load(file)
    
    if not data:
        raise ValueError(f"No data found in {filename}.")

    np_data = np.array(data)
    min_values = np.min(np_data, axis=0)

    return min_values


# Calculate median values from JSON files
median_lower_bound = calculate_median_from_json('lower_bounds.json')
median_upper_bound = calculate_median_from_json('upper_bounds.json')

print(f"Median lower HSV bound: {median_lower_bound}")
print(f"Median upper HSV bound: {median_upper_bound}")

print(f"Max value for upper hsv: {find_max_from_json('upper_bounds.json')}")
print(f"Min value for lower hsv: {find_min_from_json('lower_bounds.json')}")


#lower_bound, upper_bound = detect_green_hsv_bounds(image_path)
#print(f"Detected lower HSV bound: {lower_bound}")
#print(f"Detected upper HSV bound: {upper_bound}")






#car_center = find_car(image_path)
#print(f'The center of the car is at: {car_center[:2]}')
#print(f'The angle of the car is: {car_center[2]} degrees')