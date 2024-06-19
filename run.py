import cv2
import numpy as np
import math
#from picture.autocalibratecolors import *
from algorithm.control import *
from algorithm.utils import *


class Camera2:
    def __init__(self):
        self.hsv_ranges = {
            'red': (np.array([0, 199, 234]), np.array([12, 255, 255])), 
            'white': (np.array([  9,   0, 253]), np.array([57,75,255])), 
            'orange': (np.array([13, 187, 246]), np.array([19, 255, 255])), 
            'blue_LED': (np.array([121,  42, 248]), np.array([167, 255, 255])), 
            'LED': (np.array([111,  31,  36]), np.array([161, 253, 255]))
        }

            # 'red': (np.array([0, 160, 0]), np.array([10, 255, 255])),
            # 'white': (np.array([0, 0, 253]), np.array([57, 112, 255])),
            # 'orange': (np.array([13, 186, 250]), np.array([180, 255, 255])),
            # 'blue_LED': (np.array([121, 0, 254]), np.array([180, 255, 255])),
            # 'LED': (np.array([85, 0, 250]), np.array([180, 255, 255]))
            # {
            #     'red': (array([  0,  94, 245]), array([  9, 173, 255])), 
            #     'white': (array([  0,   0, 253]), array([ 57, 112, 255])), 
            #     'orange': (array([ 13, 125, 250]), array([180, 255, 255])), 
            #     'blue_LED': (array([110, 132, 231]), array([180, 252, 255])), 
            #     'LED': (array([ 47,   0, 250]), array([152,  59, 255]))}
        
        self.morph = True
        self.morphed_frame = None
        self.frame = None
        self.cross_lines = None
        self.white_ball_centers = []
        self.blocked_ball_centers = []
        self.egg_center = None
        self.orange_blob_centers = []
        self.blocked_orange_blobs = []
        self.blue_LED_centers = []
        self.LED_centers = []
        self.last_cross_angle = 0
        self.M = None
        self.last_valid_points = None
        self.morph_points = None
        self.robot_center = None
        self.robot_direction = None
        self.angle_to_closest_ball = None
        self.distance_to_closest_ball = None
        self.angle_to_closest_waypoint = None
        self.distance_to_closest_waypoint = None
        self.arena_dimensions = (180, 120)  # (width, height) in cm
        self.waypoint_for_closest_white_ball = None
        self.waypoint_for_closest_orange_ball = None
        self.waypoint_distance = 20  # distance from ball center to waypoint in cm

    def equalize_histogram(self, hsv_frame):
        h, s, v = cv2.split(hsv_frame)
        v = cv2.equalizeHist(v)
        return cv2.merge([h, s, v])

    def mask_and_find_contours(self, image, color, erode=False, open=False, close=False):
        hsv_lower, hsv_upper = self.hsv_ranges[color]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = self.equalize_histogram(hsv)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        # cv2.imwrite("output_mask.jpg", mask)
        # cv2.imwrite("output_frame.jpg", image)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
        if open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        if erode:
            mask = cv2.erode(mask, kernel2, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # debug_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(debug_image, contours, -1, (0, 0, 255), 2)
        # cv2.imwrite("debug_all_contours.jpg", debug_image)
        return contours, mask

    def find_sharpest_corners(self, contour, num_corners=4, epsilon_factor=0.02):
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx_corners) > num_corners:
            epsilon_factor += 0.01
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx_corners = cv2.approxPolyDP(contour, epsilon, True)
        return approx_corners.reshape(-1, 2).astype(int) if len(approx_corners) == num_corners else None

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        return rect

    def percentage_deviation(self, new_points, old_points):
        return np.linalg.norm(new_points - old_points) / np.linalg.norm(old_points) * 100

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        print("Rectangle = ", rect)
        if self.last_valid_points is not None:
            deviation = self.percentage_deviation(rect, self.last_valid_points)
            if deviation > 20:
                print(
                    f"Rejected morph points due to excessive deviation: {deviation:.2f}%")
                if self.M is not None:
                    self.morphed_frame = cv2.warpPerspective(
                        image, self.M, (image.shape[1], image.shape[0]))
                return False
        
        
        self.last_valid_points = rect
        self.morph_points = tuple(map(tuple, rect))

        # Compute width and height from the ordered points
        widthA = np.linalg.norm(rect[2] - rect[3])
        widthB = np.linalg.norm(rect[1] - rect[0])
        maxWidth = int(max(widthA, widthB))

        # Calculate height using the known aspect ratio
        height = int(
            maxWidth * self.arena_dimensions[1] / self.arena_dimensions[0])

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, height - 1],
            [0, height - 1]
        ], dtype="float32")

        self.M = cv2.getPerspectiveTransform(rect, dst)
        self.morphed_frame = cv2.warpPerspective(
            image, self.M, (maxWidth, height))
        return True

    def distance_to_cross(self, points, cx, cy, angle):
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        return np.sum(np.minimum(
            np.abs(cos_angle * (points[:, 0] - cx) +
                   sin_angle * (points[:, 1] - cy)),
            np.abs(-sin_angle * (points[:, 0] - cx) +
                   cos_angle * (points[:, 1] - cy))
        )**2)

    def fit_rotated_cross_to_contour(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)
        center = np.mean(box, axis=0)
        scaled_box = center + 2 * (box - center)
        scaled_box = np.array(scaled_box, dtype=np.int0)
        self.cross_lines = [(tuple(scaled_box[0]), tuple(
            scaled_box[2])), (tuple(scaled_box[1]), tuple(scaled_box[3]))]

    def sort_contours_by_length(self, contours, min_length=10, reverse=True):
        sorted_contours = sorted(
            contours, key=lambda c: cv2.arcLength(c, True), reverse=reverse)
        return [contour for contour in sorted_contours if cv2.arcLength(contour, True) >= min_length]

    def find_centers_in_contour_list(self, contours):
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                centers.append((int(M['m10'] / M['m00']),
                               int(M['m01'] / M['m00'])))
        return centers

    def find_white_blobs(self):
        target_frame = self.morphed_frame if self.morph else self.frame
        contours, _ = self.mask_and_find_contours(
            target_frame, color='white', erode=True, open=True)
        
        sorted_contours = self.sort_contours_by_length(
            contours, min_length=10, reverse=True)
        if not sorted_contours:
            print("No contours found.")
            self.white_ball_centers = []
            self.blocked_ball_centers = []
            self.egg_center = None
            return False

        egg_contour = sorted_contours[0]
        self.egg_center = self.find_centers_in_contour_list([egg_contour])[0]

        white_ball_contours = sorted_contours[1:11] if len(
            sorted_contours) > 1 else []
        self.white_ball_centers = self.find_centers_in_contour_list(
            white_ball_contours)

        if self.LED_centers:
            LED_center = np.mean(np.array(self.LED_centers), axis=0)
            LED_radius = np.mean(
                [np.linalg.norm(np.array(center) - LED_center) for center in self.LED_centers])
            self.white_ball_centers = [center for center in self.white_ball_centers if np.linalg.norm(
                np.array(center) - LED_center) > LED_radius]

        if self.robot_center is not None:
            self.white_ball_centers = sorted(self.white_ball_centers, key=lambda center: np.linalg.norm(
                np.array(center) - self.robot_center))

        if self.egg_center is not None and self.white_ball_centers:
            exclusion_radius = 10
            self.white_ball_centers = [center for center in self.white_ball_centers if np.linalg.norm(
                np.array(center) - self.egg_center) > exclusion_radius]

        self.blocked_ball_centers = []
        if self.robot_center is not None and self.cross_lines:
            self.white_ball_centers, self.blocked_ball_centers = self.filter_blocked_balls(
                self.white_ball_centers)

        # If orange ball is not blocked, add it to the front of the list
        if self.orange_blob_centers:
            self.white_ball_centers = self.orange_blob_centers + self.white_ball_centers

        morph_frame_width = max(
            self.morphed_frame.shape[0], self.morphed_frame.shape[1])
        r = self.waypoint_distance * \
            morph_frame_width / self.arena_dimensions[0]
        self.waypoint_for_closest_white_ball = self.calculate_waypoint(
            self.white_ball_centers[0], r)

        if self.robot_center is not None and self.white_ball_centers:
            nearest_ball_center = self.white_ball_centers[0]
            self.angle_to_closest_ball = self.calculate_angle_to_ball(
                self.robot_center, nearest_ball_center, self.robot_direction)

            self.distance_to_closest_ball = np.linalg.norm(
                np.array(nearest_ball_center) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width

        if self.robot_center is not None and self.waypoint_for_closest_white_ball:
            nearest_waypoint = self.waypoint_for_closest_white_ball[0]
            self.angle_to_closest_waypoint = self.calculate_angle_to_ball(
                self.robot_center, nearest_waypoint[0], self.robot_direction)

            self.distance_to_closest_waypoint = np.linalg.norm(
                np.array(nearest_waypoint[0]) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width

        return True

    def filter_blocked_balls(self, ball_centers):
        robot_center = np.array(self.robot_center)
        unblocked_ball_centers = []
        blocked_ball_centers = []

        for center in ball_centers:
            center = np.array(center)
            blocked = False
            for line in self.cross_lines:
                line_start = np.array(line[0])
                line_end = np.array(line[1])
                if self.intersect_lines(robot_center, center, line_start, line_end):
                    blocked_ball_centers.append(tuple(center))
                    blocked = True
                    break
            if not blocked:
                unblocked_ball_centers.append(tuple(center))

        return unblocked_ball_centers, blocked_ball_centers

    def intersect_lines(self, p1, p2, q1, q2):
        def orientation(p, q, r):
            val = (float(q[1] - p[1]) * (r[0] - q[0])) - \
                (float(q[0] - p[0]) * (r[1] - q[1]))
            return 1 if val > 0 else 2 if val < 0 else 0

        o1, o2 = orientation(p1, p2, q1), orientation(p1, p2, q2)
        o3, o4 = orientation(q1, q2, p1), orientation(q1, q2, p2)
        return o1 != o2 and o3 != o4

    def calculate_angle_to_ball(self, robot_center, ball_center, robot_direction):
        robot_center, ball_center = np.array(
            robot_center), np.array(ball_center)

        if robot_direction is None or len(robot_direction) != 2:
            raise ValueError("Robot direction is not defined correctly")

        to_ball_vector = ball_center - robot_center
        robot_direction_norm = robot_direction / \
            np.linalg.norm(robot_direction)
        to_ball_vector_norm = to_ball_vector / np.linalg.norm(to_ball_vector)

        dot_product = np.dot(robot_direction_norm, to_ball_vector_norm)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)
        cross_product = np.cross(robot_direction_norm, to_ball_vector_norm)
        if cross_product < 0:
            angle = -angle

        return np.degrees(angle)

    def calculate_waypoint(self, ball_center, r):
        if not ball_center or self.robot_center is None:
            return []

        waypoints = []

        def distance_to_line_segment(point, line):
            px, py = point
            (x1, y1), (x2, y2) = line
            v1 = np.array([px - x1, py - y1])
            v2 = np.array([x2 - x1, y2 - y1])
            projection = np.dot(v1, v2) / np.dot(v2, v2)

            if projection < 0:
                closest_point = np.array([x1, y1])
            elif projection > 1:
                closest_point = np.array([x2, y2])
            else:
                closest_point = np.array([x1, y1]) + projection * v2

            return np.linalg.norm(np.array([px, py]) - closest_point)

        def distance_to_boundaries(point, width, height):
            x, y = point
            return min(x, width - x, y, height - y)

        def generate_candidate_waypoints(ball_center, radius, num_points=100):
            cx, cy = ball_center
            candidates = []
            for i in range(num_points):
                angle = 2 * math.pi / num_points * i
                candidate = (cx + radius * math.cos(angle),
                             cy + radius * math.sin(angle))
                if 0 <= candidate[0] < img_width and 0 <= candidate[1] < img_height:
                    candidates.append(candidate)
            return candidates

        img_height, img_width = self.morphed_frame.shape[:2]

        candidates = generate_candidate_waypoints(ball_center, r)
        if not candidates:
            print(f"No valid waypoints for ball at {ball_center}")
            return waypoints

        max_min_distance = 0
        best_waypoint = None

        for candidate in candidates:
            min_distance = float('inf')

            for line in self.cross_lines:
                distance = distance_to_line_segment(candidate, line)
                min_distance = min(min_distance, distance)

            boundary_distance = distance_to_boundaries(
                candidate, img_width, img_height)
            min_distance = min(min_distance, boundary_distance)

            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_waypoint = candidate

        if best_waypoint:
            best_waypoint = (int(best_waypoint[0]), int(best_waypoint[1]))
            waypoints.append((best_waypoint, ball_center))

        return waypoints

    def find_blobs(self, color, num_points):
        target_frame = self.morphed_frame if self.morph else self.frame
        contours, _ = self.mask_and_find_contours(
            target_frame, color=color, close=True)
        
        sorted_contours = self.sort_contours_by_length(
            contours, min_length=5, reverse=True)

        if not sorted_contours:
            return False
        
        centers = self.find_centers_in_contour_list(
            sorted_contours[:num_points])
        if color == 'orange':
            self.orange_blob_centers = centers
            # Check if orange blob is blocked
            if self.robot_center is not None and self.cross_lines:
                self.orange_blob_centers, self.blocked_orange_blobs = self.filter_blocked_balls(
                    self.orange_blob_centers)
                if self.blocked_orange_blobs:
                    print("Orange blob is blocked.")

        elif color == 'blue_LED':
            self.blue_LED_centers = centers
        elif color == 'LED':
            self.LED_centers = centers
        return bool(centers)

    def find_robot(self):
        if self.LED_centers:
            self.robot_center = np.mean(self.LED_centers, axis=0)

        if len(self.LED_centers) >= 4 and self.blue_LED_centers:
            led_centers_array = np.array(self.LED_centers)
            blue_led_center = np.array(self.blue_LED_centers[0])

            sorted_indices = np.argsort(np.linalg.norm(
                led_centers_array - blue_led_center, axis=1))
            sorted_led_centers = led_centers_array[sorted_indices]

            b1, b2, f1, f2 = sorted_led_centers[0], sorted_led_centers[2], sorted_led_centers[1], sorted_led_centers[3]

            back_center = (b1 + b2) / 2
            front_center = (f1 + f2) / 2

            direction = front_center - back_center
            self.robot_direction = direction
        else:
            self.robot_direction = None

    def process_frame(self):
        try:
            #self.preprocess_frame()
            if self.morph:
                contours, _ = self.mask_and_find_contours(
                    self.frame, color='red', close=True, open=False, erode=False) #open=False, erode=False
                sorted_contours = self.sort_contours_by_length(
                    contours, min_length=50, reverse=True)
                
                if len(sorted_contours) > 2:
                    arena_contour, cross_contour = sorted_contours[1], sorted_contours[2]
                    corners = self.find_sharpest_corners(arena_contour)
                    if corners is not None and len(corners) == 4:
                        corners = np.array([corner.ravel()
                                           for corner in corners], dtype="float32")
                        if not self.four_point_transform(self.frame, corners):
                            print("Skipping frame due to invalid morph points.")
                            return
                        if cross_contour is not None:
                            transformed_contour = cv2.perspectiveTransform(
                                cross_contour.reshape(-1, 1, 2).astype(np.float32), self.M).astype(int)
                            self.fit_rotated_cross_to_contour(
                                transformed_contour)

                        self.find_blobs('LED', num_points=4)
                        self.find_blobs('orange', num_points=1)
                        self.find_blobs('blue_LED', num_points=1)
                        self.find_robot()
                        self.find_white_blobs()

                        self.draw_detected_features()
            else:
                self.morphed_frame = self.frame.copy()
                contours, _ = self.mask_and_find_contours(
                    self.frame, color='red')
                sorted_contours = self.sort_contours_by_length(
                    contours, min_length=50, reverse=True)
                if len(sorted_contours) > 2:
                    cross_contour = sorted_contours[2]
                    self.fit_rotated_cross_to_contour(cross_contour)
                self.find_blobs('LED', num_points=4)
                self.find_blobs('orange', num_points=1)
                self.find_blobs('blue_LED', num_points=1)
                self.find_robot()
                self.find_white_blobs()
                self.draw_detected_features()
        except Exception as e:
            print(f"Error processing frame: {e}")

    def draw_detected_features(self):
        if self.white_ball_centers:
            for center in self.white_ball_centers:
                cv2.circle(self.morphed_frame, tuple(
                    map(int, center)), 5, (0, 255, 0), -1)

            if self.waypoint_for_closest_white_ball:
                nearest_waypoint = self.waypoint_for_closest_white_ball[0]
                waypoint_coord = tuple(map(int, nearest_waypoint[0]))
                ball_center = tuple(map(int, nearest_waypoint[1]))

                cv2.circle(self.morphed_frame, waypoint_coord,
                           5, (0, 255, 255), -1)

                cv2.line(self.morphed_frame, waypoint_coord,
                         ball_center, (0, 255, 255), 2)

                if self.robot_center is not None:
                    cv2.line(self.morphed_frame, tuple(
                        map(int, self.robot_center)), waypoint_coord, (0, 255, 255), 2)

        if self.blocked_ball_centers:
            for center in self.blocked_ball_centers:
                cv2.circle(self.morphed_frame, tuple(
                    map(int, center)), 5, (0, 0, 0), -1)
        if self.blocked_orange_blobs:
            for center in self.blocked_orange_blobs:
                cv2.circle(self.morphed_frame, tuple(
                    map(int, center)), 5, (0, 0, 0), -1)
            cv2.circle(self.morphed_frame, tuple(
                map(int, self.blocked_orange_blobs[0])), 20, (0, 140, 255), 3)

        if self.egg_center is not None:
            cv2.circle(self.morphed_frame, tuple(
                map(int, self.egg_center)), 5, (0, 0, 255), -1)

        if self.cross_lines:
            for line in self.cross_lines:
                cv2.line(self.morphed_frame, tuple(map(int, line[0])), tuple(
                    map(int, line[1])), (0, 0, 255), 5)

        if self.orange_blob_centers:
            for center in self.orange_blob_centers:
                cv2.circle(self.morphed_frame, tuple(
                    map(int, center)), 5, (0, 255, 0), -1)
            cv2.circle(self.morphed_frame, tuple(
                map(int, self.orange_blob_centers[0])), 20, (0, 140, 255), 3)

        if self.LED_centers:
            for center in self.LED_centers:
                cv2.circle(self.morphed_frame, tuple(
                    map(int, center)), 8, (255, 255, 255), -1)

        if self.robot_center is not None:
            center = tuple(map(int, self.robot_center))
            cv2.circle(self.morphed_frame, center, 5, (255, 0, 0), -1)
            if self.robot_direction is not None:
                end_points = (center, tuple(
                    map(int, self.robot_center + 50 * self.robot_direction)))
                cv2.arrowedLine(self.morphed_frame,
                                end_points[0], end_points[1], (255, 0, 0), 2)

        if self.last_valid_points is not None:
            for pt in self.last_valid_points:
                pt = tuple(map(int, pt))
                cv2.circle(self.frame, pt, 5, (0, 0, 0), -1)

        if self.robot_center is not None:
            cv2.putText(self.morphed_frame, f"{self.angle_to_closest_waypoint:.1f} deg",
                        tuple(map(int, self.robot_center)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        if self.waypoint_for_closest_white_ball:
            nearest_waypoint = self.waypoint_for_closest_white_ball[0]
            cv2.putText(self.morphed_frame, f"{self.distance_to_closest_waypoint:.1f} cm",
                        tuple(map(int, nearest_waypoint[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        h, w = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        return cv2.resize(image, dim, interpolation=inter)

    def preprocess_frame(self):
        self.frame = cv2.GaussianBlur(self.frame, (5, 5), 0)

    def move_to_targetv6(self):
        print("HSV-ranges: ",self.hsv_ranges)
        #time.sleep(5)
        return
        #we're going for a waypoint:
        if(self.angle_to_closest_waypoint > 5 or self.angle_to_closest_waypoint < 5):
            if  self.angle_to_closest_waypoint > 100 or self.angle_to_closest_waypoint < 100:
                angle_correction = 0.4
            elif self.angle_to_closest_waypoint > 50 or self.angle_to_closest_waypoint < 50:
                angle_correction = 0.25
            else:
                angle_correction = 0.11
            if self.angle_to_closest_waypoint > 0:
                publish_controller_data((0, 0, angle_correction, 0, 0))  # Tilt right
            else:
                publish_controller_data((0, 0, (-1 * angle_correction), 0, 0))  # Tilt left
            return
        if(self.distance_to_closest_waypoint > 10):   
            if self.distance_to_closest_waypoint > 800:
                forward_speed = 0.5
            elif self.distance_to_closest_waypoint > 500:
                forward_speed = 0.3
            else:
                forward_speed = 0.15
            publish_controller_data((0, forward_speed, 0, 0, 0))
            return
        
        
        #now we expect that the robot is on the waypoint!
        if(self.angle_to_closest_ball > 5 or self.angle_to_closest_ball < 5):
            if  self.angle_to_closest_ball > 100 or self.angle_to_closest_ball < 100:
                angle_correction = 0.4
            elif self.angle_to_closest_ball > 50 or self.angle_to_closest_ball < 50:
                angle_correction = 0.25
            else:
                angle_correction = 0.11
            if self.angle_to_closest_ball > 0:
                publish_controller_data((0, 0, angle_correction, 0, 0))  # Tilt right
            else:
                publish_controller_data((0, 0, (-1 * angle_correction), 0, 0))  # Tilt left
            return
        if(self.distance_to_closest_ball > 160):   
            forward_speed = 0.12
            publish_controller_data((0, forward_speed, 0, 0, 0))
            return
        publish_controller_data((0, 0, 0, 1, 0))
        
    def start_video_stream(self, video_source, morph=True, record=False):
        self.morph = morph
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Unable to open video source {video_source}")
            return

        first_valid_points_obtained = False
        out = None  # Initialize video writer as None
        
        while True:
            ret, self.frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from video source")
                break

            #self.frame = self.resize_with_aspect_ratio(self.frame, width=640)

            if self.morph and not first_valid_points_obtained:
                for _ in range(10):
                    ret, self.frame = cap.read()
               
                # Load the HSV ranges from the file
                # loaded_hsv_ranges = np.load('hsv_ranges.npz')
                # self.hsvranges = {key: (loaded_hsv_ranges[key][0], loaded_hsv_ranges[key][1]) for key in loaded_hsv_ranges}
                self.process_frame()
                print("First set of valid points obtained.")
                first_valid_points_obtained = True
                for pt in self.last_valid_points:
                    pt = tuple(map(int, pt))
                    cv2.circle(self.frame, pt, 5, (0, 0, 0), -1)
                cv2.imshow('Initial Frame', self.frame)
                key = cv2.waitKey(0) & 0xFF
                if key == ord('r'):
                    first_valid_points_obtained = False
                    continue
                elif key == ord('a'):
                    cv2.destroyWindow('Initial Frame')
                    continue
                elif key == ord('q'):
                    break
            else:
                self.process_frame()
                cv2.imshow('Processed Frame', self.morphed_frame)
                cv2.imshow('Original Frame', self.frame)
                self.move_to_targetv6()

                # Initialize video writer with dynamic frame size
                if out is None and record:
                    frame_height, frame_width = self.morphed_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(
                        'processed_output.mp4', fourcc, 40.0, (frame_width, frame_height))
                    if not out.isOpened():
                        print("Error: Unable to open video writer")
                        break
                    else:
                        print(
                            f"Video writer initialized: {frame_width}x{frame_height}")

                # Write the processed frame to the video file
                if self.morphed_frame is not None and record:
                    out.write(self.morphed_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    def calibrate_color(self, color, video_path=None):
        def nothing(x):
            pass

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Unable to open video source {video_path}")
            return

        for _ in range(10):
            ret, self.frame = cap.read()
        #self.frame = self.resize_with_aspect_ratio(self.frame, width=640)
        #self.preprocess_frame()
        if not ret:
            print("Error: Unable to read frame from video source")
            return

        cv2.namedWindow('Calibration')
        hsv_lower, hsv_upper = self.hsv_ranges[color]
        cv2.createTrackbar('H Lower', 'Calibration',
                           hsv_lower[0], 255, nothing)
        cv2.createTrackbar('S Lower', 'Calibration',
                           hsv_lower[1], 255, nothing)
        cv2.createTrackbar('V Lower', 'Calibration',
                           hsv_lower[2], 255, nothing)
        cv2.createTrackbar('H Upper', 'Calibration',
                           hsv_upper[0], 255, nothing)
        cv2.createTrackbar('S Upper', 'Calibration',
                           hsv_upper[1], 255, nothing)
        cv2.createTrackbar('V Upper', 'Calibration',
                           hsv_upper[2], 255, nothing)

        while True:
            h_lower = cv2.getTrackbarPos('H Lower', 'Calibration')
            s_lower = cv2.getTrackbarPos('S Lower', 'Calibration')
            v_lower = cv2.getTrackbarPos('V Lower', 'Calibration')
            h_upper = cv2.getTrackbarPos('H Upper', 'Calibration')
            s_upper = cv2.getTrackbarPos('S Upper', 'Calibration')
            v_upper = cv2.getTrackbarPos('V Upper', 'Calibration')
            lower_hsv = np.array([h_lower, s_lower, v_lower])
            upper_hsv = np.array([h_upper, s_upper, v_upper])
            hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
            hsv = self.equalize_histogram(hsv)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
            cv2.imshow(f'Original Frame {color}', self.frame)
            cv2.imshow(f'Binary Mask for {color}', mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                self.hsv_ranges[color] = (lower_hsv, upper_hsv)
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = Camera2()
    video_path = 1
    camera.calibrate_color("red",video_path)
    camera.calibrate_color("white",video_path)
    camera.calibrate_color("orange",video_path)
    camera.calibrate_color("blue_LED",video_path)
    camera.calibrate_color("LED",video_path)
    print("hsv_ranges: ", camera.hsv_ranges)
    camera.start_video_stream(video_path, morph=True, record=False)


