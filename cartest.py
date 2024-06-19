import cv2
import numpy as np
import math

class Car:
    def __init__(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle

class Camera:
    def __init__(self):
        self.hsv_ranges = {
            'blue_LED': (np.array([121, 0, 254]), np.array([180, 255, 255])),
            'LED': (np.array([85, 0, 250]), np.array([180, 255, 255]))
        }
        self.LED_centers = []
        self.blue_LED_centers = []
        self.robot_center = None
        self.robot_direction = None

    def equalize_histogram(self, hsv_frame):
        h, s, v = cv2.split(hsv_frame)
        v = cv2.equalizeHist(v)
        return cv2.merge([h, s, v])

    def mask_and_find_contours(self, image, color):
        hsv_lower, hsv_upper = self.hsv_ranges[color]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = self.equalize_histogram(hsv)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask

    def find_centers_in_contour_list(self, contours):
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                centers.append((int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])))
        return centers

    def sort_contours_by_length(self, contours, min_length=10, reverse=True):
        sorted_contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=reverse)
        return [contour for contour in sorted_contours if cv2.arcLength(contour, True) >= min_length]

    def find_blobs(self, image, color, num_points):
        contours, mask = self.mask_and_find_contours(image, color=color)
        sorted_contours = self.sort_contours_by_length(contours, min_length=10, reverse=True)
        if not sorted_contours:
            return False, None
        centers = self.find_centers_in_contour_list(sorted_contours[:num_points])
        if color == 'blue_LED':
            self.blue_LED_centers = centers
        elif color == 'LED':
            self.LED_centers = centers
        return bool(centers), contours

    def find_robot(self):
        if self.LED_centers:
            self.robot_center = np.mean(self.LED_centers, axis=0)

        if len(self.LED_centers) >= 4 and self.blue_LED_centers:
            led_centers_array = np.array(self.LED_centers)
            blue_led_center = np.array(self.blue_LED_centers[0])

            sorted_indices = np.argsort(np.linalg.norm(led_centers_array - blue_led_center, axis=1))
            sorted_led_centers = led_centers_array[sorted_indices]

            b1, b2, f1, f2 = sorted_led_centers[0], sorted_led_centers[2], sorted_led_centers[1], sorted_led_centers[3]

            back_center = (b1 + b2) / 2
            front_center = (f1 + f2) / 2

            direction = front_center - back_center
            self.robot_direction = direction
        else:
            self.robot_direction = None

    def calculate_angle(self):
        if self.robot_direction is not None:
            angle = math.degrees(math.atan2(self.robot_direction[1], self.robot_direction[0]))
            return angle
        return 0

    def process_image(self, image):
        valid_leds, led_contours = self.find_blobs(image, 'LED', num_points=4)
        valid_blue_led, blue_led_contours = self.find_blobs(image, 'blue_LED', num_points=1)
        self.find_robot()

        output_image = image.copy()
        if valid_leds:
            cv2.drawContours(output_image, led_contours, -1, (255, 0, 0), 2)
        if valid_blue_led:
            cv2.drawContours(output_image, blue_led_contours, -1, (0, 0, 255), 2)

        if self.robot_center is not None and self.robot_direction is not None:
            x, y = map(int, self.robot_center)
            angle = self.calculate_angle()
            car = Car(x, y, angle)

            # Draw robot center
            cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)

            # Draw direction arrow
            arrow_length = 50
            end_point = (int(x + arrow_length * math.cos(math.radians(angle))),
                         int(y + arrow_length * math.sin(math.radians(angle))))
            cv2.arrowedLine(output_image, (x, y), end_point, (0, 255, 0), 2)

            return car, output_image
        return None, output_image


# Example usage
if __name__ == "__main__":
    image_path = "extra/test\images\WIN_20240619_12_11_45_Pro.jpg"
    image = cv2.imread(image_path)
    camera = Camera()
    car, output_image = camera.process_image(image)

    if car:
        print(f"Car Position: x={car.x}, y={car.y}, angle={car.angle}")
    else:
        print("Car not detected")

    cv2.imshow('LED Contours and Car Direction', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
