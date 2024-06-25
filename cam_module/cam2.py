import copy
from re import T
import cv2
import numpy as np
import math
from time import sleep
from multiprocessing import Queue
import queue


class Camera2:
    def __init__(self, control_flags):
        self.hsv_ranges = {
            'red': (np.array([87, 81, 138]), np.array([180, 255, 255])),
            'white': (np.array([0, 0, 253]), np.array([180, 59, 255])),
            'orange': (np.array([0, 68, 139]), np.array([9, 255, 255])),
            # 'blue': (np.array([90, 50, 0]), np.array([122, 255, 255]))
            'blue': (np.array([27, 47, 0]), np.array([106, 255, 255])),
            'green': (np.array([163, 40, 255]), np.array([180, 255, 255]))
        }

        # Camera properties and frame data
        self.cap = None
        self.morphed_frame = None
        self.frame = None

        # White ball properties
        self.white_ball_centers = []
        self.blocked_white_centers = []
        self.angle_to_closest_white_ball = None
        self.distance_to_closest_white_ball = None
        self.angle_to_closest_white_waypoint = None
        self.distance_to_closest_white_waypoint = None
        self.waypoint_for_closest_white_ball = None
        self.vector_to_white_ball_robot_frame = []
        self.vector_to_white_waypoint_robot_frame = []

        # Egg properties
        self.egg_center = None
        self.egg_size = 0
        self.egg_area = 0
        self.egg_scale_factor = 1.5

        # Orange blob properties
        self.orange_blob_area = 4
        self.orange_blob_detected = False

        self.orange_blob_centers = []
        self.blocked_orange_centers = []
        self.angle_to_closest_orange_ball = None
        self.distance_to_closest_orange_ball = None
        self.angle_to_closest_orange_waypoint = None
        self.distance_to_closest_orange_waypoint = None
        self.waypoint_for_closest_orange_ball = None
        self.vector_to_orange_ball_robot_frame = []
        self.vector_to_orange_waypoint_robot_frame = []

        # Robot properties
        self.robot_radius_scale_factor = 1.5
        self.green_centers = []
        self.blue_centers = []
        self.robot_center = None
        self.robot_radius = None
        self.robot_direction = None
        self.robot_critical_length = 25  # cm
        self.robot_center_coor = [0, 0]  # pixels
        self.robot_angle_coor = 0  # degrees
        self.blue_centers = []

        self.robot_center_calibrated = False
        self.robot_angle_calibrated = False
        self.robot_critical_length_calibrated = False

        # cross properties
        self.last_cross_angle = 0
        self.cross_lines = None

        # Arena properties
        self.morph = True
        self.M = None
        self.last_valid_points = None
        self.morph_points = None
        self.arena_dimensions = (166.8, 122)  # (width, height) in cm
        self.waypoint_distance = 30  # distance from ball center to waypoint in cm
        self.corner_tolerance = 20  # percentage deviation tolerance for morph points
        self.arena_waypoints = []
        self.arena_data = []
        self.manual_corners = True

        # Control flags
        self.control_flags = control_flags

    def get_data(self):
        """
        Extracts and returns relevant data from the camera class.
        This method can be customized to return any required class attributes.
        """
        return {
            # White ball data
            'white_ball_centers': self.white_ball_centers,
            'blocked_white_centers': self.blocked_white_centers,
            'angle_to_closest_white_ball': self.angle_to_closest_white_ball,
            'distance_to_closest_white_ball': self.distance_to_closest_white_ball,
            'angle_to_closest_white_waypoint': self.angle_to_closest_white_waypoint,
            'distance_to_closest_white_waypoint': self.distance_to_closest_white_waypoint,
            'waypoint_for_closest_white_ball': self.waypoint_for_closest_white_ball,
            'vector_to_white_ball_robot_frame': self.vector_to_white_ball_robot_frame,
            'vector_to_white_waypoint_robot_frame': self.vector_to_white_waypoint_robot_frame,

            # Orange ball data
            'orange_blob_detected': self.orange_blob_detected,
            'orange_blob_centers': self.orange_blob_centers,
            'blocked_orange_centers': self.blocked_orange_centers,
            'angle_to_closest_orange_ball': self.angle_to_closest_orange_ball,
            'distance_to_closest_orange_ball': self.distance_to_closest_orange_ball,
            'angle_to_closest_orange_waypoint': self.angle_to_closest_orange_waypoint,
            'distance_to_closest_orange_waypoint': self.distance_to_closest_orange_waypoint,
            'waypoint_for_closest_orange_ball': self.waypoint_for_closest_orange_ball,
            'vector_to_orange_ball_robot_frame': self.vector_to_orange_ball_robot_frame,
            'vector_to_orange_waypoint_robot_frame': self.vector_to_orange_waypoint_robot_frame,

            # Robot data
            'robot_critical_length': self.robot_critical_length,

            # Arena data
            'arena_data': self.arena_data,

        }

    def find_arena_waypoints(self):
        frame = self.morphed_frame
        width, height = frame.shape[:2]

        # Define the goals and arena points
        goal_1 = (height / 2, 0)
        goal_2 = (height / 2, width)
        arena_points = [
            (0, 0), (0, width / 2), (0, width),
            (height, 0), (height, width / 2), (height, width)
        ]

        # Combine goals and arena points
        combined_points = [goal_1] + arena_points + [goal_2]

        # map combined points to lsit of int tuples
        combined_points = [tuple(map(int, point)) for point in combined_points]

        r = self.waypoint_distance * \
            max(self.morphed_frame.shape[0],
                self.morphed_frame.shape[1]) / self.arena_dimensions[0]

        self.arena_waypoints = []
        for point in combined_points:
            waypoint = self.calculate_waypoint(point, r)
            if waypoint:
                # Store only the waypoint, not the original point
                self.arena_waypoints.append(waypoint[0])

    def equalize_histogram(self, hsv_frame):
        h, s, v = cv2.split(hsv_frame)
        v = cv2.equalizeHist(v)
        return cv2.merge([h, s, v])

    def mask_and_find_contours(self, image, color, erode=False, open=False, close=False):
        hsv_lower, hsv_upper = self.hsv_ranges[color]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = self.equalize_histogram(hsv)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        if open:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        if close:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
        if erode:
            mask = cv2.erode(mask, kernel2, iterations=1)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, hierarchy

    def find_sharpest_corners_1(self, contour, num_corners=4):
        try:
            # Create a blank mask with the same size as the frame
            mask = np.zeros_like(self.frame, dtype=np.uint8)

            # Draw the contour on the mask
            cv2.drawContours(mask, [contour], -1,
                             (255, 255, 255), thickness=cv2.FILLED)

            # Convert the mask to a single channel (grayscale)
            if len(mask.shape) > 2:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            # Use cv2.goodFeaturesToTrack to find the sharpest corners
            corners = cv2.goodFeaturesToTrack(
                mask,
                maxCorners=num_corners,
                qualityLevel=0.01,
                minDistance=10
            )

            if corners is not None:
                # Refine corner points to sub-pixel accuracy
                refined_corners = cv2.cornerSubPix(
                    mask,
                    corners,
                    (5, 5),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS +
                              cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                )

                return refined_corners.reshape(-1, 2).astype(int)
            else:
                print("No corners found")
                return None
        except Exception as e:
            print(f"Error finding corners: {e}")
            return None

    def find_sharpest_corners_2(self, contour, num_corners=4, epsilon_factor=0.02):
        try:
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx_corners = cv2.approxPolyDP(contour, epsilon, True)

            while len(approx_corners) > num_corners:
                epsilon_factor += 0.01
                epsilon = epsilon_factor * cv2.arcLength(contour, True)
                approx_corners = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx_corners) == num_corners:
                approx_corners = approx_corners.reshape(
                    -1, 2).astype(np.float32)

                # Convert the frame to grayscale
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                # Refine corner points to sub-pixel accuracy
                refined_corners = cv2.cornerSubPix(
                    gray,
                    approx_corners,
                    (5, 5),
                    (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS +
                              cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
                )

                return refined_corners.astype(int)
            else:
                return None
        except Exception as e:
            print(f"Error finding corners: {e}")
            return None

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
        if self.last_valid_points is not None:
            deviation = self.percentage_deviation(rect, self.last_valid_points)
            if deviation > self.corner_tolerance:
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
        try:
            self.morphed_frame = cv2.warpPerspective(
                image, self.M, (maxWidth, height))
        except:
            print("Error warping perspective")
            return False
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
        scagreen_box = center + 2 * (box - center)
        scagreen_box = np.array(scagreen_box, dtype=np.intp)
        self.cross_lines = [(tuple(scagreen_box[0]), tuple(
            scagreen_box[2])), (tuple(scagreen_box[1]), tuple(scagreen_box[3]))]

    def sort_contours_by_arch_length(self, contours, min_length=10, reverse=True):
        sorted_contours = sorted(
            contours, key=lambda c: cv2.arcLength(c, True), reverse=reverse)
        return [contour for contour in sorted_contours if cv2.arcLength(contour, True) >= min_length]

    def sort_contours_by_bounding_box_area(self, contours, min_area=10, reverse=True):
        # Sort contours based on the area of the bounding box
        sorted_contours = sorted(
            contours, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3], reverse=reverse)

        # Filter contours based on the minimum bounding box area
        return [contour for contour in sorted_contours if cv2.boundingRect(contour)[2] * cv2.boundingRect(contour)[3] >= min_area]

    def sort_contours_by_min_area_rect(self, contours, min_area=10, reverse=True):
        # Sort contours based on the area of the minimum area bounding rectangle
        sorted_contours = sorted(
            contours, key=lambda c: cv2.minAreaRect(c)[1][0] * cv2.minAreaRect(c)[1][1], reverse=reverse
        )

        # Filter contours based on the minimum area of the bounding rectangle
        return [contour for contour in sorted_contours if cv2.minAreaRect(contour)[1][0] * cv2.minAreaRect(contour)[1][1] >= min_area]

    def sort_contours_by_area(self, contours, min_area=10, reverse=True):
        # Sort contours based on the area
        sorted_contours = sorted(
            contours, key=cv2.contourArea, reverse=reverse)

        # Filter contours based on the minimum area
        filtered_contours = [
            contour for contour in sorted_contours if cv2.contourArea(contour) >= min_area]

        return filtered_contours

    def find_centers_in_contour_list(self, contours):
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            if M['m00'] != 0:
                centers.append((int(M['m10'] / M['m00']),
                               int(M['m01'] / M['m00'])))
        return centers

    def find_white_blobs(self):
        try:
            target_frame = self.morphed_frame if self.morph else self.frame

            # Find contours in the target frame
            contours, _ = self.mask_and_find_contours(
                target_frame, color='white', erode=False, open=False, close=False)

            # Sort contours by length
            sorted_contours = self.sort_contours_by_area(
                contours, min_area=5, reverse=True)

            if not sorted_contours:
                print("No contours found.")
                self.reset_ball_centers()
                return False

            # Identify the egg contour and its properties
            egg_contour = sorted_contours[0]
            self.egg_center = self.find_centers_in_contour_list([egg_contour])[
                0]

            # find egg area from min area rect
            self.egg_area = cv2.minAreaRect(
                egg_contour)[1][0] * cv2.minAreaRect(egg_contour)[1][1]

            self.white_ball_centers = self.find_centers_in_contour_list(
                sorted_contours[1:11] if len(sorted_contours) > 1 else [])

            print(f"White ball centers: {self.white_ball_centers}")
            # append orange center to white ball centers
            if self.orange_blob_centers:
                self.white_ball_centers.append(self.orange_blob_centers[0])

            # Remove balls within robot radius of robot center
            if self.robot_center is not None and self.robot_radius is not None:
                self.white_ball_centers = [center for center in self.white_ball_centers if np.linalg.norm(
                    np.array(center) - self.robot_center) > self.robot_radius_scale_factor * self.robot_radius]

            # Sort white balls by their distance to the robot center
            if self.robot_center is not None:
                self.white_ball_centers = sorted(self.white_ball_centers, key=lambda center: np.linalg.norm(
                    np.array(center) - self.robot_center))

            # Exclude balls within the exclusion radius of the egg
            if self.egg_center is not None:
                exclusion_radius = self.egg_scale_factor * self.egg_size
                self.white_ball_centers = [center for center in self.white_ball_centers if np.linalg.norm(
                    np.array(center) - self.egg_center) > exclusion_radius]

            # Initialize blocked_white_centers list
            self.blocked_white_centers = []

            # Filter out blocked balls based on crossing lines
            if self.robot_center is not None and self.cross_lines:
                self.white_ball_centers, self.blocked_white_centers = self.filter_blocked_balls(
                    self.white_ball_centers)

            return True
        except Exception as e:
            print(f"Error finding white blobs: {e}")
            return False

    def reset_ball_centers(self):
        self.white_ball_centers = []
        self.blocked_white_centers = []
        self.egg_center = None

    def find_waypoint_for_closest_white_ball(self):
        print('Finding waypoint for white ball')
        r = self.waypoint_distance * \
            max(self.morphed_frame.shape[0],
                self.morphed_frame.shape[1]) / self.arena_dimensions[0]
        while self.white_ball_centers:
            self.waypoint_for_closest_white_ball = self.calculate_waypoint(
                self.white_ball_centers[0], r)
            temp_white_centers, temp_blocked_centers = self.filter_blocked_balls(
                [self.white_ball_centers[0]])
            if not temp_blocked_centers:
                break
            else:
                blocked_center = self.white_ball_centers.pop(0)
                self.blocked_white_centers.append(blocked_center)
        if not self.white_ball_centers:
            self.waypoint_for_closest_white_ball = None

    def find_waypoint_for_closest_orange_ball(self):
        r = self.waypoint_distance * \
            max(self.morphed_frame.shape[0],
                self.morphed_frame.shape[1]) / self.arena_dimensions[0]
        print('Finding waypoint for orange ball')
        while self.orange_blob_centers:
            self.waypoint_for_closest_orange_ball = self.calculate_waypoint(
                self.orange_blob_centers[0], r)
            temp_orange_centers, temp_blocked_centers = self.filter_blocked_balls(
                [self.orange_blob_centers[0]])
            if not temp_blocked_centers:
                break
            else:
                blocked_center = self.orange_blob_centers.pop(0)
                self.blocked_orange_centers.append(blocked_center)
        if not self.orange_blob_centers:
            self.waypoint_for_closest_orange_ball = None

    def calculate_vectors_to_targets(self, morph_frame_width):
        def calculate_and_store_vectors(ball_center, vector_attr, angle_attr, distance_attr):
            angle = self.calculate_angle_to_ball(
                self.robot_center, ball_center, self.robot_direction)
            distance = np.linalg.norm(np.array(
                ball_center) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
            vector = self.calculate_vector_to_target_in_robot_frame(
                ball_center)
            vector = self.normalize_vector(vector)
            setattr(self, vector_attr, vector)
            setattr(self, angle_attr, angle)
            setattr(self, distance_attr, distance)
            return distance, angle

        if self.robot_center is not None:
            if self.white_ball_centers:
                nearest_ball_center = self.white_ball_centers[0]
                calculate_and_store_vectors(
                    nearest_ball_center,
                    'vector_to_white_ball_robot_frame',
                    'angle_to_closest_white_ball',
                    'distance_to_closest_white_ball'
                )

            if self.orange_blob_centers:
                nearest_orange_ball_center = self.orange_blob_centers[0]
                calculate_and_store_vectors(
                    nearest_orange_ball_center,
                    'vector_to_orange_ball_robot_frame',
                    'angle_to_closest_orange_ball',
                    'distance_to_closest_orange_ball'
                )

            if self.waypoint_for_closest_white_ball:
                nearest_waypoint = self.waypoint_for_closest_white_ball[0]
                calculate_and_store_vectors(
                    nearest_waypoint[0],
                    'vector_to_white_waypoint_robot_frame',
                    'angle_to_closest_white_waypoint',
                    'distance_to_closest_white_waypoint'
                )

            if self.waypoint_for_closest_orange_ball:
                nearest_waypoint = self.waypoint_for_closest_orange_ball[0]
                calculate_and_store_vectors(
                    nearest_waypoint[0],
                    'vector_to_orange_waypoint_robot_frame',
                    'angle_to_closest_orange_waypoint',
                    'distance_to_closest_orange_waypoint'
                )

            # Calculate and store vectors for all arena waypoints
            if self.arena_waypoints is not None:
                self.arena_data = []
                for waypoint in self.arena_waypoints:
                    waypoint_coords = waypoint[0]
                    arena_point = waypoint[1]
                    vector = self.calculate_vector_to_target_in_robot_frame(
                        waypoint_coords)
                    vector = self.normalize_vector(vector)
                    distance = np.linalg.norm(np.array(
                        waypoint_coords) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
                    angle = self.calculate_angle_to_ball(
                        self.robot_center, arena_point, self.robot_direction)
                    self.arena_data.append([vector, distance, angle])
            else:
                print('No arena waypoints found')

    # def calculate_vectors_to_targets(self, morph_frame_width):
    #     def calculate_and_store_vectors(ball_center, vector_attr, angle_attr, distance_attr):
    #         angle = self.calculate_angle_to_ball(
    #             self.robot_center, ball_center, self.robot_direction)
    #         distance = np.linalg.norm(np.array(
    #             ball_center) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #         vector = self.calculate_vector_to_target_in_robot_frame(
    #             ball_center)
    #         vector = self.normalize_vector(vector)
    #         setattr(self, vector_attr, vector)
    #         setattr(self, angle_attr, angle)
    #         setattr(self, distance_attr, distance)
    #         return distance, angle

    #     if self.robot_center is not None:
    #         if self.white_ball_centers:
    #             nearest_ball_center = self.white_ball_centers[0]
    #             calculate_and_store_vectors(
    #                 nearest_ball_center,
    #                 'vector_to_white_ball_robot_frame',
    #                 'angle_to_closest_white_ball',
    #                 'distance_to_closest_white_ball'
    #             )

    #         if self.orange_blob_centers:
    #             nearest_orange_ball_center = self.orange_blob_centers[0]
    #             calculate_and_store_vectors(
    #                 nearest_orange_ball_center,
    #                 'vector_to_orange_ball_robot_frame',
    #                 'angle_to_closest_orange_ball',
    #                 'distance_to_closest_orange_ball'
    #             )

    #         if self.waypoint_for_closest_white_ball:
    #             nearest_waypoint = self.waypoint_for_closest_white_ball[0]
    #             calculate_and_store_vectors(
    #                 nearest_waypoint[0],
    #                 'vector_to_white_waypoint_robot_frame',
    #                 'angle_to_closest_white_waypoint',
    #                 'distance_to_closest_white_waypoint'
    #             )

    #         if self.waypoint_for_closest_orange_ball:
    #             nearest_waypoint = self.waypoint_for_closest_orange_ball[0]
    #             calculate_and_store_vectors(
    #                 nearest_waypoint[0],
    #                 'vector_to_orange_waypoint_robot_frame',
    #                 'angle_to_closest_orange_waypoint',
    #                 'distance_to_closest_orange_waypoint'
    #             )

    #         # Calculate and store vectors for all arena waypoints
    #         if self.arena_waypoints is not None:
    #             temp_list = []
    #             for waypoint in self.arena_waypoints:
    #                 element = []
    #                 vector = self.calculate_vector_to_target_in_robot_frame(
    #                     waypoint[0])
    #                 vector = self.normalize_vector(vector)
    #                 element.append(vector)
    #                 distance = np.linalg.norm(np.array(
    #                     waypoint[0]) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #                 element.append(distance)
    #                 angle = self.calculate_angle_to_ball(
    #                     self.robot_center, waypoint[1], self.robot_direction)
    #                 element.append(angle)
    #                 temp_list.append(element)

    #             self.arena_data = temp_list
    #         else:
    #             print('No arena waypoints found')

    # def calculate_vectors_to_targets(self, morph_frame_width):
    #     def calculate_and_store_vectors(ball_center, vector_attr, angle_attr, distance_attr):
    #         angle = self.calculate_angle_to_ball(
    #             self.robot_center, ball_center, self.robot_direction)
    #         distance = np.linalg.norm(np.array(
    #             ball_center) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #         vector = self.calculate_vector_to_target_in_robot_frame(
    #             ball_center)
    #         vector = self.normalize_vector(vector)
    #         setattr(self, vector_attr, vector)
    #         setattr(self, angle_attr, angle)
    #         setattr(self, distance_attr, distance)

    #     if self.robot_center is not None:
    #         if self.white_ball_centers:
    #             nearest_ball_center = self.white_ball_centers[0]
    #             calculate_and_store_vectors(
    #                 nearest_ball_center,
    #                 'vector_to_white_ball_robot_frame',
    #                 'angle_to_closest_white_ball',
    #                 'distance_to_closest_white_ball'
    #             )

    #         if self.orange_blob_centers:
    #             nearest_orange_ball_center = self.orange_blob_centers[0]
    #             calculate_and_store_vectors(
    #                 nearest_orange_ball_center,
    #                 'vector_to_orange_ball_robot_frame',
    #                 'angle_to_closest_orange_ball',
    #                 'distance_to_closest_orange_ball'
    #             )

    #         if self.waypoint_for_closest_white_ball:
    #             nearest_waypoint = self.waypoint_for_closest_white_ball[0]
    #             calculate_and_store_vectors(
    #                 nearest_waypoint[0],
    #                 'vector_to_white_waypoint_robot_frame',
    #                 'angle_to_closest_white_waypoint',
    #                 'distance_to_closest_white_waypoint'
    #             )

    #         if self.waypoint_for_closest_orange_ball:
    #             nearest_waypoint = self.waypoint_for_closest_orange_ball[0]
    #             calculate_and_store_vectors(
    #                 nearest_waypoint[0],
    #                 'vector_to_orange_waypoint_robot_frame',
    #                 'angle_to_closest_orange_waypoint',
    #                 'distance_to_closest_orange_waypoint'
    #             )

    #         # Calculate and store vectors for all arena waypoints
    #         if self.arena_waypoints:
    #             self.arena_waypoint_vectors = []
    #             self.arena_waypoint_distances = []
    #             for waypoint, _ in self.arena_waypoints:
    #                 vector = self.calculate_vector_to_target_in_robot_frame(
    #                     waypoint)
    #                 vector = self.normalize_vector(vector)
    #                 distance = np.linalg.norm(np.array(
    #                     waypoint) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #                 self.arena_waypoint_vectors.append(vector)
    #                 self.arena_waypoint_distances.append(distance)

    # def calculate_vectors_to_targets(self, morph_frame_width):

    #     # Calculate the angle and distance to the closest white ball
    #     if self.robot_center is not None and self.white_ball_centers:
    #         nearest_ball_center = self.white_ball_centers[0]
    #         self.angle_to_closest_white_ball = self.calculate_angle_to_ball(
    #             self.robot_center, nearest_ball_center, self.robot_direction)
    #         self.distance_to_closest_white_ball = np.linalg.norm(np.array(
    #             nearest_ball_center) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #         self.vector_to_white_ball_robot_frame = self.calculate_vector_to_target_in_robot_frame(
    #             nearest_ball_center)
    #         self.vector_to_white_ball_robot_frame = self.normalize_vector(
    #             self.vector_to_white_ball_robot_frame)

    #     # Calculate the angle and distance to the closest orange ball
    #     if self.robot_center is not None and self.orange_blob_centers:
    #         nearest_orange_ball_center = self.orange_blob_centers[0]
    #         self.angle_to_closest_orange_ball = self.calculate_angle_to_ball(
    #             self.robot_center, nearest_orange_ball_center, self.robot_direction)
    #         self.distance_to_closest_orange_ball = np.linalg.norm(np.array(
    #             nearest_orange_ball_center) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #         self.vector_to_orange_ball_robot_frame = self.calculate_vector_to_target_in_robot_frame(
    #             nearest_orange_ball_center)
    #         self.vector_to_orange_ball_robot_frame = self.normalize_vector(
    #             self.vector_to_orange_ball_robot_frame)

    #     # Calculate the angle and distance to the closest waypoint
    #     if self.robot_center is not None and self.waypoint_for_closest_white_ball:
    #         nearest_waypoint = self.waypoint_for_closest_white_ball[0]
    #         self.angle_to_closest_white_waypoint = self.calculate_angle_to_ball(
    #             self.robot_center, nearest_waypoint[0], self.robot_direction)
    #         self.distance_to_closest_white_waypoint = np.linalg.norm(np.array(
    #             nearest_waypoint[0]) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #         self.vector_to_white_waypoint_robot_frame = self.calculate_vector_to_target_in_robot_frame(
    #             nearest_waypoint[0])
    #         self.vector_to_white_waypoint_robot_frame = self.normalize_vector(
    #             self.vector_to_white_waypoint_robot_frame)

    #     # do the same for waypoint to orange ball
    #     if self.robot_center is not None and self.waypoint_for_closest_orange_ball:
    #         nearest_waypoint = self.waypoint_for_closest_orange_ball[0]
    #         self.angle_to_closest_orange_waypoint = self.calculate_angle_to_ball(
    #             self.robot_center, nearest_waypoint[0], self.robot_direction)
    #         self.distance_to_closest_orange_waypoint = np.linalg.norm(np.array(
    #             nearest_waypoint[0]) - self.robot_center) * self.arena_dimensions[0] / morph_frame_width
    #         self.vector_to_orange_waypoint_robot_frame = self.calculate_vector_to_target_in_robot_frame(
    #             nearest_waypoint[0])
    #         self.vector_to_orange_waypoint_robot_frame = self.normalize_vector(
    #             self.vector_to_orange_waypoint_robot_frame)

    def calculate_vector_to_target_in_robot_frame(self, target):
        # Calculate the vector from the robot center to the target in global coordinates
        vector_to_target_global = np.array(target) - self.robot_center

        # Calculate the angle (theta) of the robot's direction
        theta = np.arctan2(self.robot_direction[1], self.robot_direction[0])

        # Create a rotation matrix to rotate the global vector into the robot's frame
        rotation_matrix = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # Rotate the global vector to get the vector in the robot's frame
        return np.dot(rotation_matrix.T, vector_to_target_global)

    def normalize_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def filter_blocked_balls(self, ball_centers):
        robot_center = np.array(self.robot_center)
        unblocked_white_centers = []
        blocked_white_centers = []

        for center in ball_centers:
            center = np.array(center)
            blocked = False
            for line in self.cross_lines:
                line_start = np.array(line[0])
                line_end = np.array(line[1])
                if self.intersect_lines(robot_center, center, line_start, line_end):
                    blocked_white_centers.append(tuple(center))
                    blocked = True
                    break
            if not blocked:
                unblocked_white_centers.append(tuple(center))

        return unblocked_white_centers, blocked_white_centers

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
        print(f"Calculating waypoint for ball at {ball_center}")
        if not ball_center:
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

        def generate_candidate_waypoints(ball_center, radius, num_points=400):
            cx, cy = ball_center
            candidates = []
            for i in range(num_points):
                angle = 2 * math.pi / num_points * i
                candidate = (cx + radius * math.cos(angle),
                             cy + radius * math.sin(angle))
                candidate = (int(candidate[0]), int(candidate[1]))
                # Ensure candidates are within the frame boundaries, adjust if they are at the edge
                candidate = (
                    min(max(candidate[0], 0), img_width - 1),
                    min(max(candidate[1], 0), img_height - 1)
                )
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
            waypoints.append((best_waypoint, ball_center))

        print(f"Waypoint for ball at {ball_center}: {waypoints}")
        return waypoints

    # def calculate_waypoint(self, ball_center, r):
    #     print(f"Calculating waypoint for ball at {ball_center}")
    #     if not ball_center:
    #         return []

    #     waypoints = []

    #     def distance_to_line_segment(point, line):
    #         px, py = point
    #         (x1, y1), (x2, y2) = line
    #         v1 = np.array([px - x1, py - y1])
    #         v2 = np.array([x2 - x1, y2 - y1])
    #         projection = np.dot(v1, v2) / np.dot(v2, v2)

    #         if projection < 0:
    #             closest_point = np.array([x1, y1])
    #         elif projection > 1:
    #             closest_point = np.array([x2, y2])
    #         else:
    #             closest_point = np.array([x1, y1]) + projection * v2

    #         return np.linalg.norm(np.array([px, py]) - closest_point)

    #     def distance_to_boundaries(point, width, height):
    #         x, y = point
    #         return min(x, width - x, y, height - y)

    #     def generate_candidate_waypoints(ball_center, radius, num_points=400):
    #         cx, cy = ball_center
    #         candidates = []
    #         for i in range(num_points):
    #             angle = 2 * math.pi / num_points * i
    #             candidate = (cx + radius * math.cos(angle),
    #                          cy + radius * math.sin(angle))
    #             if 0 <= candidate[0] < img_width and 0 <= candidate[1] < img_height:
    #                 candidates.append(candidate)
    #         return candidates

    #     img_height, img_width = self.morphed_frame.shape[:2]

    #     candidates = generate_candidate_waypoints(ball_center, r)
    #     if not candidates:
    #         print(f"No valid waypoints for ball at {ball_center}")
    #         return waypoints

    #     max_min_distance = 0
    #     best_waypoint = None

    #     for candidate in candidates:
    #         min_distance = float('inf')

    #         for line in self.cross_lines:
    #             distance = distance_to_line_segment(candidate, line)
    #             min_distance = min(min_distance, distance)

    #         boundary_distance = distance_to_boundaries(
    #             candidate, img_width, img_height)
    #         min_distance = min(min_distance, boundary_distance)

    #         if min_distance > max_min_distance:
    #             max_min_distance = min_distance
    #             best_waypoint = candidate

    #     if best_waypoint:
    #         best_waypoint = (int(best_waypoint[0]), int(best_waypoint[1]))
    #         waypoints.append((best_waypoint, ball_center))

    #     print(f"Waypoint for ball at {ball_center}: {waypoints}")
    #     return waypoints

    def find_blobs(self, color, num_points):
        target_frame = self.morphed_frame if self.morph else self.frame
        contours, _ = self.mask_and_find_contours(
            target_frame, color=color, close=False, open=False, erode=False)

        sorted_contours = self.sort_contours_by_area(
            contours, min_area=1, reverse=True)

        if not sorted_contours:
            return False

        # Find the centers of the top num_points contours
        centers = self.find_centers_in_contour_list(
            sorted_contours[:num_points])

        if color == 'orange':
            self.orange_blob_detected = True
            self.orange_blob_centers = centers
            # Check if orange blob is blocked
            if self.robot_center is not None and self.cross_lines:
                self.orange_blob_centers, self.blocked_orange_centers = self.filter_blocked_balls(
                    self.orange_blob_centers)

            # get orange blob area from bounding rect
            if len(sorted_contours) > 0:
                bounding_rect = cv2.boundingRect(sorted_contours[0])
                self.orange_blob_area = bounding_rect[2] * bounding_rect[3]
            else:
                self.orange_blob_detected = False

        elif color == 'blue':
            # Find the robot center and radious from the green centers
            self.blue_centers = centers
            self.robot_center = np.mean(np.array(centers), axis=0)

            # add the robot center correction and convert to int
            self.robot_center = self.robot_center + \
                np.array(self.robot_center_coor)
            self.robot_center = tuple(map(int, self.robot_center))

            self.robot_radius = np.mean(
                [np.linalg.norm(np.array(center) - self.robot_center) for center in centers])
            self.blue_centers = centers

        elif color == 'green':
            self.green_centers = centers
            self.find_robot_direction()

        return bool(centers)

    def find_robot_direction(self):
        try:
            print("Finding robot direction.")
            if len(self.blue_centers) == 4:
                blue_centers_array = np.array(self.blue_centers)

                # sort the blue centers by distance to green center
                blue_centers_array = sorted(blue_centers_array, key=lambda center: np.linalg.norm(
                    np.array(center) - self.green_centers[0]))

                front_point1, front_point2 = blue_centers_array[0], blue_centers_array[1]
                back_point1, back_point2 = blue_centers_array[2], blue_centers_array[3]

                front_center = (front_point1 + front_point2) / 2
                back_center = (back_point1 + back_point2) / 2

                direction = front_center - back_center

                print(f"Direction: {direction}")

                # Normalize the direction vector
                self.robot_direction = direction / np.linalg.norm(direction)

                # Convert angle correction from degrees to radians
                angle_correction_radians = np.radians(self.robot_angle_coor)

                # Correct the robot direction
                self.robot_direction = np.dot(
                    np.array([[np.cos(angle_correction_radians), -np.sin(angle_correction_radians)],
                              [np.sin(angle_correction_radians), np.cos(angle_correction_radians)]]),
                    self.robot_direction
                )
        except Exception as e:
            print(f"Error finding robot direction: {e}")

    def resize_frame(self, frame, width=None, height=None, inter=cv2.INTER_AREA):
        # initialize the dimensions of the frame to be resized and
        # grab the image size
        dim = None
        (h, w) = frame.shape[:2]

        # if both the width and height are None, then return the
        # original frame
        if width is None and height is None:
            return frame

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the frame
        resized = cv2.resize(frame, dim, interpolation=inter)

        # return the resized frame
        return resized

    def process_frame(self):
        try:
            self.preprocess_frame()
            if self.morph:
                # if self.update_arena is True, the arena corners and cross are updated

                # if self.update_arena:
                if self.control_flags['update_arena']:
                    print("Updating arena corners and cross.")

                    if not self.manual_corners:
                        contours, hierarchy = self.mask_and_find_contours(
                            self.frame, color='red', close=False, open=False, erode=False)

                        sorted_contours = self.sort_contours_by_area(
                            contours, min_area=10, reverse=True)

                        if len(sorted_contours) > 2:

                            arena_contour, cross_contour = sorted_contours[1], sorted_contours[2]

                            corners = self.find_sharpest_corners_2(
                                arena_contour)

                            if corners is not None and len(corners) == 4:
                                corners = np.array([corner.ravel()
                                                    for corner in corners], dtype="float32")

                                self.four_point_transform(self.frame, corners)

                                if cross_contour is not None:
                                    transformed_contour = cv2.perspectiveTransform(
                                        cross_contour.reshape(-1, 1, 2).astype(np.float32), self.M).astype(int)
                                    self.fit_rotated_cross_to_contour(
                                        transformed_contour)

                                # calculate arena waypoints
                                # print("Calculating arena waypoints.")
                                if self.morphed_frame is not None:
                                    self.find_arena_waypoints()

                                # print(f"Arena waypoints: {self.arena_waypoints}")

                            else:
                                print("Failed to find corners.")

                    else:
                        self.four_point_transform(
                            self.frame, self.last_valid_points)
                        contours, hierarchy = self.mask_and_find_contours(
                            self.frame, color='red', close=False, open=False, erode=False)
                        sorted_contours = self.sort_contours_by_area(
                            contours, min_area=50, reverse=True)
                        cross_contour = sorted_contours[0]
                        self.fit_rotated_cross_to_contour(cross_contour)

                else:
                    self.four_point_transform(
                        self.frame, self.last_valid_points)

                if self.morphed_frame is not None:

                    if self.control_flags['update_robot']:
                        try:
                            print("Finding blue centers.")
                            self.find_blobs('blue', num_points=4)
                            print(f'Blue centers: {self.blue_centers}')

                            print("Finding green centers.")
                            self.find_blobs('green', num_points=1)
                            print(f'Green centers: {self.green_centers}')
                        except Exception as e:
                            print(f"Failed to find robot: {e}")

                    if self.control_flags['update_orange_balls']:
                        try:
                            self.find_blobs('orange', num_points=1)
                            self.find_waypoint_for_closest_orange_ball()
                        except Exception as e:
                            print(f"Error finding orange blobs: {e}")

                    if self.control_flags['update_white_balls']:
                        try:
                            self.find_white_blobs()
                            self.find_waypoint_for_closest_white_ball()
                        except Exception as e:
                            print(f"Error finding white blobs: {e}")

                    if self.control_flags['update_robot']:
                        try:
                            print("Calculating vectors to targets.")
                            self.calculate_vectors_to_targets(
                                morph_frame_width=max(self.morphed_frame.shape[0], self.morphed_frame.shape[1]))
                            # print arena waypoints
                            print(f"Arena data: {self.arena_data}")
                        except Exception as e:
                            print(f"Error calculating vectors: {e}")

            self.draw_detected_features()
        except Exception as e:
            print(f"Error processing frame: {e}")

    def draw_detected_features(self):

        # Draw white ball centers
        try:
            if self.white_ball_centers is not None:
                for center in self.white_ball_centers:
                    cv2.circle(self.morphed_frame, center, 5, (0, 255, 0), -1)

                if self.waypoint_for_closest_white_ball:
                    nearest_waypoint = self.waypoint_for_closest_white_ball[0]

                    waypoint_coord = nearest_waypoint[0]
                    ball_center = nearest_waypoint[1]

                    cv2.circle(self.morphed_frame, waypoint_coord,
                               5, (0, 255, 255), -1)

                    cv2.line(self.morphed_frame, waypoint_coord,
                             ball_center, (0, 255, 255), 2)

                    if self.robot_center is not None:
                        cv2.line(self.morphed_frame, self.robot_center,
                                 waypoint_coord, (0, 255, 255), 2)

                    #  draw vector to waypoint
                    # if self.vector_to_white_waypoint_robot_frame is not None:
                    #     scale_factor = 50
                    #     end_point = tuple(map(int, np.array(
                    #         waypoint_coord) + scale_factor * self.vector_to_white_waypoint_robot_frame))
                    #     cv2.arrowedLine(
                    #         self.morphed_frame, waypoint_coord, end_point, (0, 255, 255), 2)
        except Exception as e:
            print(f"Error drawing white blobs: {e}")

        try:  # Draw blocked centers
            if self.blocked_white_centers:
                for center in self.blocked_white_centers:
                    cv2.circle(self.morphed_frame, center, 5, (0, 0, 0), -1)

            if self.blocked_orange_centers:
                for center in self.blocked_orange_centers:
                    cv2.circle(self.morphed_frame, center, 5, (0, 0, 0), -1)
        except Exception as e:
            print(f"Error drawing blocked centers: {e}")

        try:  # Draw egg center
            if self.egg_center is not None:
                cv2.circle(self.morphed_frame,
                           self.egg_center, 5, (0, 0, 255), -1)
        except Exception as e:
            print(f"Error drawing egg center: {e}")

        try:  # Draw cross lines
            if self.cross_lines:
                for line in self.cross_lines:
                    cv2.line(self.morphed_frame,
                             line[0], line[1], (0, 0, 255), 5)

        except Exception as e:
            print(f"Error drawing cross lines: {e}")

        # try:  # Draw orange ball centers
        #     if self.orange_blob_centers is not None:
        #         for center in self.orange_blob_centers:
        #             cv2.circle(self.morphed_frame, center, 5, (0, 255, 0), -1)

        #         if self.waypoint_for_closest_orange_ball:
        #             nearest_waypoint = self.waypoint_for_closest_orange_ball[0]

        #             waypoint_coord = nearest_waypoint[0]
        #             ball_center = nearest_waypoint[1]

        #             cv2.circle(self.morphed_frame, waypoint_coord,
        #                        5, (0, 255, 255), -1)

        #             # cv2.line(self.morphed_frame, waypoint_coord,
        #             #          ball_center, (0, 255, 255), 2)

        #             # if self.robot_center is not None:
        #             #     cv2.line(self.morphed_frame, self.robot_center,
        #             #              waypoint_coord, (0, 255, 255), 2)
        # except Exception as e:
        #     print(f"Error drawing orange blobs: {e}")

        try:  # Draw robot center and direction
            if self.robot_center is not None:
                center = self.robot_center
                cv2.circle(self.morphed_frame, center, 5, (255, 0, 0), -1)

            if self.robot_direction is not None:

                # scale the robot direction vector by the critical length converted to pixels
                scale_factor = self.robot_critical_length * \
                    max(self.morphed_frame.shape[0],
                        self.morphed_frame.shape[1]) / self.arena_dimensions[0]

                end_points = (center, tuple(
                    map(int, self.robot_center + scale_factor * self.robot_direction)))

                cv2.arrowedLine(self.morphed_frame,
                                end_points[0], end_points[1], (255, 0, 0), 2)

        except Exception as e:
            print(f"Error drawing robot: {e}")

        try:  # Draw last valid cornor points
            if self.last_valid_points is not None:
                for pt in self.last_valid_points:
                    pt = tuple(map(int, pt))
                    cv2.circle(self.frame, pt, 5, (0, 0, 0), -1)
        except Exception as e:
            print(f"Error drawing last valid points: {e}")

        try:  # Draw text

            # Draw the angle to the closest white ball
            if self.robot_center is not None:
                cv2.putText(self.morphed_frame, f"{self.angle_to_closest_white_ball:.1f} deg",
                            self.robot_center, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Draw the distance to the closest white ball
            if self.waypoint_for_closest_white_ball is not None:
                nearest_waypoint = self.waypoint_for_closest_white_ball[0]
                cv2.putText(self.morphed_frame, f"{self.distance_to_closest_white_waypoint:.1f} cm",
                            nearest_waypoint[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error drawing text: {e}")

        # try to draw arena waypoints
        try:
            if self.arena_waypoints is not None:
                for waypoint in self.arena_waypoints:
                    cv2.circle(self.morphed_frame,
                               waypoint[0], 5, (255, 0, 255), -1)
        except:
            print('Error drawing arena waypoints')

        # # Draw arena data
        # try:
        #     if self.arena_data is not None:
        #         for index_number, data in enumerate(self.arena_data):
        #             vector = data[0]
        #             distance = data[1]
        #             angle = data[2]
        #             # convert distance to pixels
        #             distance = distance * \
        #                 max(self.morphed_frame.shape[0],
        #                     self.morphed_frame.shape[1]) / self.arena_dimensions[0]

        #             # Calculate the endpoint correctly
        #             end_point = tuple(
        #                 map(int, np.array(self.robot_center) + distance * vector))

        #             # Draw arrowed line from robot center to end point
        #             cv2.arrowedLine(self.morphed_frame, tuple(
        #                 map(int, self.robot_center)), end_point, (255, 0, 255), 2)

        #             # Draw index number at waypoint
        #             cv2.putText(self.morphed_frame, f"{index_number}", end_point,
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        # except Exception as e:
        #     print(f'Error drawing arena data: {e}')

    # draw blue centers
        try:
            if self.blue_centers is not None:
                for center in self.blue_centers:
                    cv2.circle(self.morphed_frame, center, 5, (255, 0, 0), -1)
        except Exception as e:
            print(f"Error drawing blue centers: {e}")

    # draw green centers
        try:
            if self.green_centers is not None:
                for center in self.green_centers:
                    cv2.circle(self.morphed_frame, center, 5, (0, 255, 0), -1)
        except Exception as e:
            print(f"Error drawing green centers: {e}")

    def preprocess_frame(self):
        return
        self.frame = cv2.GaussianBlur(self.frame, (3, 3), 0)

    def start_video_stream(self, video_source, queue=None, morph=True, record=False, resize=None, manual_corners=False):
        print("Starting video stream...")
        self.morph = morph
        self.manual_corners = manual_corners

        if self.cap is None:
            print(f"Opening video source {video_source}")
            self.cap = cv2.VideoCapture(video_source, cv2.CAP_V4L2)
            if not self.cap.isOpened():
                print(f"Error: Unable to open video source {video_source}")
                return
            self.cap.set(cv2.CAP_PROP_FPS, 40)
            sleep(5)  # Allow camera to adjust to lighting conditions

        first_valid_points_obtained = False
        out = None  # Initialize video writer as None

        temp_flags = self.control_flags.copy()

        while True:
            print('Entering first while loop try block')
            try:
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Error: Unable to read frame from video source")
                    break

                if resize:
                    self.frame = self.resize_frame(self.frame, width=resize)

                if manual_corners and not first_valid_points_obtained:
                    print("Manual corner selection enabled.")
                    self.last_valid_points = self.find_corners_manual()
                    self.four_point_transform(
                        self.frame, self.last_valid_points)
                    first_valid_points_obtained = True
                    for key in self.control_flags.keys():
                        self.control_flags[key] = False
                    self.control_flags['update_robot'] = True
                    self.process_frame()
                    self.calibrate_robot()
                    continue

                if self.morph and not first_valid_points_obtained:
                    print("Obtaining initial points for morphing...")

                    ret, self.frame = self.cap.read()
                    if not ret:
                        print("Error: Unable to read frame from video source")
                        break

                    # Set all flags to True
                    print('Initial control flags:', self.control_flags)
                    print('Attempting to set all flags to true')
                    for key in self.control_flags.keys():
                        self.control_flags[key] = True
                    print('All flags set to true')
                    print('Processing initial frame')

                    self.process_frame()

                    print('Processed initial frame')

                    # Check if last_valid_points is not None
                    print('Attempting to check if last_valid_points is not None')
                    if self.last_valid_points is not None:
                        print('last_valid_points is not None')
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
                            first_valid_points_obtained = True
                            # reset all flags to false except for update_robot
                            for key in self.control_flags.keys():
                                self.control_flags[key] = False
                            self.control_flags['update_robot'] = True
                            self.calibrate_robot()
                            continue
                        elif key == ord('q'):
                            break
                    else:
                        print('last_valid_points is None')
                        cv2.imshow('Initial Frame', self.frame)
                        key = cv2.waitKey(0) & 0xFF
                        if key == ord('q'):
                            break

                # Calibrate robot
                if not (self.robot_center_calibrated and self.robot_angle_calibrated and self.robot_critical_length_calibrated):
                    self.calibrate_robot()
                    self.robot_center_calibrated = True
                    self.robot_angle_calibrated = True
                    self.robot_critical_length_calibrated = True
                    # reset the control flags to the original values
                    for key in self.control_flags.keys():
                        self.control_flags[key] = temp_flags[key]

                else:
                    self.process_frame()
                    cv2.imshow('Processed Frame', self.morphed_frame)
                    cv2.imshow('Original Frame', self.frame)

                    # Initialize video writer with dynamic frame size
                    if out is None and record:
                        frame_height, frame_width = self.morphed_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        out = cv2.VideoWriter(
                            'processed_output.mp4', fourcc, 40.0, (frame_width, frame_height))
                        if not out.isOpened():
                            print("Error: Unable to open video writer")
                            break

                    # Write the processed frame to the video file
                    if self.morphed_frame is not None and record:
                        out.write(self.morphed_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # Pause the video stream if 'p' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('p'):
                        sleep(1)
                        cv2.waitKey(-1)

                    # Send data to the queue
                    if queue is not None and self.morphed_frame is not None:
                        data = self.get_data()
                        try:
                            queue.put_nowait(data)
                        except Exception as e:
                            print(f"Error putting data in queue: {e}")

            except KeyboardInterrupt:
                print("Keyboard interrupt detected.")
                self.cap.release()
                cv2.destroyAllWindows()
                break
            except Exception as e:
                print(f"Error in start_video_stream while loop: {e}")
                break

        self.cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

    def calibrate_robot(self):
        while not (self.robot_center_calibrated and self.robot_angle_calibrated and self.robot_critical_length_calibrated):
            self.process_frame()
            cv2.imshow('Robot calibration', self.morphed_frame)

            if not self.robot_center_calibrated:
                print(
                    f'Calibrating robot center. Current center: {self.robot_center}')
                print(f'Robot center correction: {self.robot_center_coor}')
                # wait for arrow key press
                key = cv2.waitKey(0) & 0xFF

                # if up arrow key is pressed
                if key == 82:
                    self.robot_center_coor[1] += 1
                # if down arrow key is pressed
                elif key == 84:
                    self.robot_center_coor[1] -= 1
                # if left arrow key is pressed
                elif key == 81:
                    self.robot_center_coor[0] -= 1
                # if right arrow key is pressed
                elif key == 83:
                    self.robot_center_coor[0] += 1

                # accept the calibration on 'a' key press
                if key == ord('a'):
                    print('Robot center calibrated')
                    self.robot_center_calibrated = True
                    continue

                if key == ord('q'):
                    break

            elif not self.robot_angle_calibrated:
                print(
                    f"Calibrating robot angle. Current correction: {self.robot_angle_coor}")

                # wait for arrow key press
                key = cv2.waitKey(0) & 0xFF

                # if right arrow key is pressed
                if key == 83:
                    self.robot_angle_coor += 0.5

                # if left arrow key is pressed
                elif key == 81:
                    self.robot_angle_coor -= 0.5

                # accept the calibration on 'a' key press
                if key == ord('a'):
                    print('Robot angle calibrated')
                    self.robot_angle_calibrated = True
                    continue

            elif not self.robot_critical_length_calibrated:
                print(
                    f"Calibrating robot critical length. Current length: {self.robot_critical_length}")

                # wait for arrow key press
                key = cv2.waitKey(0) & 0xFF

                # if up arrow key is pressed
                if key == 82:
                    self.robot_critical_length += 1
                # if down arrow key is pressed
                elif key == 84:
                    self.robot_critical_length -= 1

                # accept the calibration on 'a' key press
                if key == ord('a'):
                    print('Robot critical length calibrated')
                    self.robot_critical_length_calibrated = True
                    continue

        cv2.destroyWindow('Robot calibration')

    def calibrate_color(self, colors, video_path=None, resize=None):
        def nothing(x):
            pass

        try:
            self.cap = cv2.VideoCapture(
                video_path, cv2.CAP_V4L2)  # , cv2.CAP_V4L2
            self.cap.set(cv2.CAP_PROP_FPS, 40)
        except Exception as e:
            print(f"Error: Unable to open video source {video_path}")
            return

        if not self.cap.isOpened():
            print(f"Error: Unable to open video source {video_path}")
            return

        # Sleep to allow camera to adjust to lighting conditions
        sleep(3)
        ret, self.frame = self.cap.read()
        if not ret:
            print("Error: Unable to read frame from video source")
            return

        if resize:
            self.frame = self.resize_frame(self.frame, width=resize)

        frame_height, frame_width = self.frame.shape[:2]

        for color in colors:
            # Create windows once
            cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
            cv2.namedWindow(f'Original Frame {color}', cv2.WINDOW_NORMAL)
            cv2.namedWindow(f'Binary Mask {color}', cv2.WINDOW_NORMAL)

            # Adjust window sizes to match the frame size
            cv2.resizeWindow(
                f'Original Frame {color}', frame_width, frame_height)
            cv2.resizeWindow(f'Binary Mask {color}', frame_width, frame_height)
            # Set a reasonable size for the calibration window
            cv2.resizeWindow('Calibration', 400, 300)

            # Create trackbars once per color
            hsv_lower, hsv_upper = self.hsv_ranges[color]
            cv2.createTrackbar('H Lower', 'Calibration',
                               int(hsv_lower[0]), 180, nothing)
            cv2.createTrackbar('S Lower', 'Calibration',
                               int(hsv_lower[1]), 255, nothing)
            cv2.createTrackbar('V Lower', 'Calibration',
                               int(hsv_lower[2]), 255, nothing)
            cv2.createTrackbar('H Upper', 'Calibration',
                               int(hsv_upper[0]), 180, nothing)
            cv2.createTrackbar('S Upper', 'Calibration',
                               int(hsv_upper[1]), 255, nothing)
            cv2.createTrackbar('V Upper', 'Calibration',
                               int(hsv_upper[2]), 255, nothing)

            while True:
                # sleep to allow the camera to adjust to lighting conditions
                ret, self.frame = self.cap.read()
                if not ret:
                    print("Error: Unable to read frame from video source")
                    break

                if resize:
                    self.frame = self.resize_frame(self.frame, width=resize)

                self.preprocess_frame()

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

                contours, _ = cv2.findContours(
                    mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                cv2.drawContours(self.frame, contours, -1, (0, 255, 0), 3)
                cv2.imshow(f'Original Frame {color}', self.frame)
                cv2.imshow(f'Binary Mask {color}', mask)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('a'):
                    print(f"Calibration for {color} completed.")
                    self.hsv_ranges[color] = (lower_hsv, upper_hsv)
                    print(f"HSV ranges for {color}: {self.hsv_ranges[color]}")
                    break
                elif key == ord('q'):

                    break

            cv2.destroyAllWindows()
        print('exiting calibration')
        # self.cap.release()

    def find_corners_manual(self):
        def draw_corners(image, corners):
            for i, corner in enumerate(corners):
                cv2.circle(image, tuple(corner), 5, (0, 255, 0), -1)
                cv2.putText(image, str(i), tuple(corner),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        frame_height, frame_width = self.frame.shape[:2]

        # Initial guess: corners close to the frame corners
        corners = [
            (20, 20),  # Top-left
            (frame_width - 20, 20),  # Top-right
            (frame_width - 20, frame_height - 20),  # Bottom-right
            (20, frame_height - 20)  # Bottom-left
        ]

        selected_corner = 0

        while True:
            frame_copy = self.frame.copy()
            draw_corners(frame_copy, corners)
            cv2.imshow("Select Corners", frame_copy)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):  # Move selected corner left
                corners[selected_corner] = (
                    corners[selected_corner][0] - 1, corners[selected_corner][1])
            elif key == ord('d'):  # Move selected corner right
                corners[selected_corner] = (
                    corners[selected_corner][0] + 1, corners[selected_corner][1])
            elif key == ord('w'):  # Move selected corner up
                corners[selected_corner] = (
                    corners[selected_corner][0], corners[selected_corner][1] - 1)
            elif key == ord('s'):  # Move selected corner down
                corners[selected_corner] = (
                    corners[selected_corner][0], corners[selected_corner][1] + 1)
            elif key == ord('n'):  # Select next corner
                selected_corner = (selected_corner + 1) % 4

        cv2.destroyWindow("Select Corners")

        # Order points: top-left, top-right, bottom-right, bottom-left
        corners = np.array(corners, dtype="float32")
        rect = self.order_points(corners)
        self.last_valid_points = rect
        return self.last_valid_points


def camera_process(queue, video_path, control_flags):
    print("Starting camera process...")
    print("Control flags:", control_flags)
    camera = Camera2(control_flags)
    colors = ['blue', 'green', 'red', 'orange', 'white']
    camera.calibrate_color(colors, video_path, resize=False)
    camera.start_video_stream(video_path, queue=queue,
                              morph=True, record=False, resize=False)


# Example usage:
if __name__ == "__main__":
    control_flags = {
        'update_orange_balls': False,
        'update_white_balls': False,
        'update_robot': True,
        'update_arena': False
    }

    camera = Camera2(control_flags=control_flags)
    video_path = '/dev/video8'
    colors = ['blue', 'green', 'red', 'orange', 'white']
    camera.calibrate_color(colors, video_path, resize=False)
    camera.start_video_stream(video_path, morph=True,
                              record=False, resize=False, manual_corners=False)
