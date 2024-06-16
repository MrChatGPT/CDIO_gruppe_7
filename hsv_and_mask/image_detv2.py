import cv2
import numpy as np
import os

def convert_rgb_to_hsv(rgb_color):
    rgb_color = np.uint8([[rgb_color]])  # Convert to a format OpenCV expects
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)
    return hsv_color[0][0]

# Example RGB values for different objects
#robot_dark_rgb = [145, 233, 173]
#robot_light_rgb = [190, 235, 194]
# white_balls_dark_rgb = [220, 208, 186]
# white_balls_light_rgb = [253, 250, 233]
orange_ball_dark_rgb = [234, 139, 45]
orange_ball_light_rgb = [255, 216, 52]
red_arena_dark_rgb = [207, 59, 31]
red_arena_light_rgb = [213, 89, 65]

# Convert the RGB values to HSV
#robot_dark_hsv = convert_rgb_to_hsv(robot_dark_rgb)
#robot_light_hsv = convert_rgb_to_hsv(robot_light_rgb)
# white_balls_dark_hsv = convert_rgb_to_hsv(white_balls_dark_rgb)
# white_balls_light_hsv = convert_rgb_to_hsv(white_balls_light_rgb)
orange_ball_dark_hsv = convert_rgb_to_hsv(orange_ball_dark_rgb)
orange_ball_light_hsv = convert_rgb_to_hsv(orange_ball_light_rgb)
red_arena_dark_hsv = convert_rgb_to_hsv(red_arena_dark_rgb)
red_arena_light_hsv = convert_rgb_to_hsv(red_arena_light_rgb)

#print("Robot Car Dark HSV:", robot_dark_hsv)
#print("Robot Car Light HSV:", robot_light_hsv)
# print("White Balls Dark HSV:", white_balls_dark_hsv)
# print("White Balls Light HSV:", white_balls_light_hsv)
print("Orange Ball Dark HSV:", orange_ball_dark_hsv)
print("Orange Ball Light HSV:", orange_ball_light_hsv)
print("Red Arena Dark HSV:", red_arena_dark_hsv)
print("Red Arena Light HSV:", red_arena_light_hsv)

def expand_hsv_range(hsv_color, h_range=10, s_range=50, v_range=50):
    lower_bound = np.array([max(hsv_color[0] - h_range, 0), max(hsv_color[1] - s_range, 0), max(hsv_color[2] - v_range, 0)])
    upper_bound = np.array([min(hsv_color[0] + h_range, 179), min(hsv_color[1] + s_range, 255), min(hsv_color[2] + v_range, 255)])
    return lower_bound, upper_bound

# Expand the HSV ranges
#robot_hsv_range = expand_hsv_range(robot_dark_hsv), expand_hsv_range(robot_light_hsv)
# white_balls_hsv_range = expand_hsv_range(white_balls_dark_hsv), expand_hsv_range(white_balls_light_hsv)
orange_ball_hsv_range = expand_hsv_range(orange_ball_dark_hsv), expand_hsv_range(orange_ball_light_hsv)
red_arena_hsv_range = expand_hsv_range(red_arena_dark_hsv), expand_hsv_range(red_arena_light_hsv)
robot_hsv_range = (np.array([44,56,141]), np.array([179,255,255]))
# Define a fixed HSV range for white color detection
white_balls_hsv_range = (np.array([0, 0, 200]), np.array([179, 50, 255]))


print("\nRobot Car HSV Range:", robot_hsv_range)
print("\nWhite Balls HSV Range:", white_balls_hsv_range)
print("\nOrange Ball HSV Range:", orange_ball_hsv_range)
print("\nRed Arena HSV Range:", red_arena_hsv_range)

def create_mask(image, lower_color, upper_color):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_image, lower_color, upper_color)
    return mask

def reduce_noise(mask):
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.medianBlur(mask, 5)
    return mask

def process_images(folder_path, hsv_ranges):
    for filename in os.listdir(folder_path):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            
            # Create masks using the provided HSV ranges
            robot_car_mask = create_mask(image, hsv_ranges["robot_car"][0], hsv_ranges["robot_car"][1])
            white_balls_mask = create_mask(image, hsv_ranges["white_balls"][0], hsv_ranges["white_balls"][1])
            orange_ball_mask = create_mask(image, hsv_ranges["orange_ball"][0][0], hsv_ranges["orange_ball"][1][1])
            red_arena_mask = create_mask(image, hsv_ranges["red_arena"][0][0], hsv_ranges["red_arena"][1][1])
            
            # Combine the orange ball and red arena masks
            combined_orange_red_arena_mask = cv2.bitwise_or(orange_ball_mask, red_arena_mask)
            
            # Reduce noise in the white balls mask
            white_balls_mask = reduce_noise(white_balls_mask)


            # Save masks
            cv2.imwrite(os.path.join(folder_path, f"{filename}_robot_car_mask.png"), robot_car_mask)
            cv2.imwrite(os.path.join(folder_path, f"{filename}_white_balls_mask.png"), white_balls_mask)
            cv2.imwrite(os.path.join(folder_path, f"{filename}_combined_orange_red_arena_mask.png"), combined_orange_red_arena_mask)
            print(f"Masks saved for {filename}")

# Use the HSV ranges calculated from the sample image
hsv_ranges = {
    "robot_car": robot_hsv_range,
    "white_balls": white_balls_hsv_range,
    "orange_ball": orange_ball_hsv_range,
    "red_arena": red_arena_hsv_range
}

# Specify the folder path containing the images
folder_path = "newest_images/"
process_images(folder_path, hsv_ranges)
