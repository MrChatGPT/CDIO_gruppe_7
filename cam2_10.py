from importlib.resources import open_binary
import cv2
import numpy as np
from scipy.optimize import minimize


class Camera2:
    def __init__(self):
        self.hsv_ranges = {
            'red': (np.array([0, 160, 0]), np.array([10, 255, 255])),
            'white': (np.array([0, 0, 253]), np.array([57, 112, 255])),
            'orange': (np.array([13, 186, 250]), np.array([180, 255, 255])),
            'blue_LED': (np.array([121, 0, 254]), np.array([180, 255, 255])),
            'LED': (np.array([85, 0, 250]), np.array([180, 255, 255]))
        }
        self.morph = True
        self.morphed_frame = None
        self.frame = None
        self.cross_lines = None
        self.white_ball_centers = []
        self.blocked_ball_centers = []
        self.egg_center = None
        self.orange_blob_centers = []
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

    def equalize_histogram(self, hsv_frame):
        h, s, v = cv2.split(hsv_frame)
        v = cv2.equalizeHist(v)
        return cv2.merge([h, s, v])

    def mask_and_find_contours(self, image, color, erode=False, open=False):
        hsv_lower, hsv_upper = self.hsv_ranges[color]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = self.equalize_histogram(hsv)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

        if open:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        if erode:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.erode(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask

    def find_sharpest_corners_method1(self, contour, num_corners=4, epsilon_factor=0.02):
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)
        while len(approx_corners) > num_corners:
            epsilon_factor += 0.01
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx_corners = cv2.approxPolyDP(contour, epsilon, True)
        return approx_corners.reshape(-1, 2).astype(int) if len(approx_corners) == num_corners else None

    def find_sharpest_corners_method2(self, mask, contour, num_corners=4):
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(
            contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        corners = cv2.goodFeaturesToTrack(
            contour_mask, maxCorners=num_corners, qualityLevel=0.01, minDistance=10)
        return corners.astype(int) if corners is not None else None

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
            if deviation > 20:
                print(
                    f"Rejected morph points due to excessive deviation: {deviation:.2f}%")
                if self.M is not None:
                    self.morphed_frame = cv2.warpPerspective(
                        image, self.M, (image.shape[1], image.shape[0]))
                return False
        self.last_valid_points = rect
        self.morph_points = tuple(map(tuple, rect))
        maxWidth = int(
            max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
        maxHeight = int(
            max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1,
                       maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        self.M = cv2.getPerspectiveTransform(rect, dst)
        self.morphed_frame = cv2.warpPerspective(
            image, self.M, (maxWidth, maxHeight))
        return True

    def distance_to_cross(self, points, cx, cy, angle):
        cos_angle, sin_angle = np.cos(angle), np.sin(angle)
        return np.sum(np.minimum(
            np.abs(cos_angle * (points[:, 0] - cx) +
                   sin_angle * (points[:, 1] - cy)),
            np.abs(-sin_angle * (points[:, 0] - cx) +
                   cos_angle * (points[:, 1] - cy))
        )**2)

    def fit_rotated_cross_to_contour_method1(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype=np.float32)

        # Find the center of the box
        center = np.mean(box, axis=0)

        # Scale the box by a factor of 2
        scaled_box = center + 2 * (box - center)

        # Convert scaled_box to integer coordinates for drawing
        scaled_box = np.array(scaled_box, dtype=np.int0)

        self.cross_lines = [(tuple(scaled_box[0]), tuple(scaled_box[2])),
                            (tuple(scaled_box[1]), tuple(scaled_box[3]))]

    def fit_rotated_cross_to_contour_method2(self, contour):
        M = cv2.moments(contour)
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] /
                                               M['m00']) if M['m00'] != 0 else (0, 0)
        contour_points = contour.reshape(-1, 2)
        max_distance = np.max(np.linalg.norm(
            contour_points - np.array([cx, cy]), axis=1))
        result = minimize(lambda angle: self.distance_to_cross(
            contour_points, cx, cy, angle), self.last_cross_angle)
        best_angle = result.x[0]
        self.last_cross_angle = best_angle
        cross_length = max_distance
        cos_angle, sin_angle = np.cos(best_angle), np.sin(best_angle)
        self.cross_lines = [
            ((int(cx + cross_length * cos_angle), int(cy + cross_length * sin_angle)),
             (int(cx - cross_length * cos_angle), int(cy - cross_length * sin_angle))),
            ((int(cx + cross_length * -sin_angle), int(cy + cross_length * cos_angle)),
             (int(cx - cross_length * -sin_angle), int(cy - cross_length * cos_angle)))
        ]

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

        # Find the egg contour and its center
        egg_contour = sorted_contours[0]
        self.egg_center = self.find_centers_in_contour_list([egg_contour])[0]

        # Find the centers of the white balls
        white_ball_contours = sorted_contours[1:11] if len(
            sorted_contours) > 1 else []
        self.white_ball_centers = self.find_centers_in_contour_list(
            white_ball_contours)

        # Exclude white blobs that lie within the area of self.LED_centers
        if self.LED_centers:
            LED_center = np.mean(np.array(self.LED_centers), axis=0)
            LED_radius = np.mean(
                [np.linalg.norm(np.array(center) - LED_center) for center in self.LED_centers])
            self.white_ball_centers = [center for center in self.white_ball_centers if np.linalg.norm(
                np.array(center) - LED_center) > LED_radius]

        # Sort white ball centers by distance to robot center
        if self.robot_center is not None:
            self.white_ball_centers = sorted(self.white_ball_centers, key=lambda center: np.linalg.norm(
                np.array(center) - self.robot_center))
            self.distance_to_closest_ball = np.linalg.norm(
                np.array(self.white_ball_centers[0]) - self.robot_center)
            print(
                f"Distance to nearest ball: {self.distance_to_closest_ball:.2f} pixels")

        # Exclude white blobs that lie within a certain radius of the egg center
        if self.egg_center is not None and self.white_ball_centers:
            exclusion_radius = 10  # Define a fixed radius for exclusion
            self.white_ball_centers = [center for center in self.white_ball_centers if np.linalg.norm(
                np.array(center) - self.egg_center) > exclusion_radius]

        # Filter out balls blocked by the cross
        self.blocked_ball_centers = []
        if self.robot_center is not None and self.cross_lines:
            self.white_ball_centers, self.blocked_ball_centers = self.filter_blocked_balls(
                self.white_ball_centers)

        # Calculate the angle to the nearest ball center
        if self.robot_center is not None and self.white_ball_centers:
            nearest_ball_center = self.white_ball_centers[0]
            self.angle_to_closest_ball = self.calculate_angle_to_ball(
                self.robot_center, nearest_ball_center, self.robot_direction)
            print(
                f"Angle to nearest ball: {self.angle_to_closest_ball:.2f} degrees")

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
        """Check if line segment p1p2 intersects with line segment q1q2 using vectorized operations."""
        def orientation(p, q, r):
            """Return the orientation of the triplet (p, q, r).
            0 -> p, q and r are collinear
            1 -> Clockwise
            2 -> Counterclockwise
            """
            val = (float(q[1] - p[1]) * (r[0] - q[0])) - \
                (float(q[0] - p[0]) * (r[1] - q[1]))
            if val > 0:
                return 1
            elif val < 0:
                return 2
            else:
                return 0

        # Check if the line segments intersect
        o1 = orientation(p1, p2, q1)
        o2 = orientation(p1, p2, q2)
        o3 = orientation(q1, q2, p1)
        o4 = orientation(q1, q2, p2)

        # General case
        if o1 != o2 and o3 != o4:
            return True

        return False

    def calculate_angle_to_ball(self, robot_center, ball_center, robot_direction):
        robot_center = np.array(robot_center)
        ball_center = np.array(ball_center)

        # Ensure robot_direction is correctly defined as a vector
        if robot_direction is None or len(robot_direction) != 2:
            raise ValueError("Robot direction is not defined correctly")

        # Vector from robot center to ball center
        to_ball_vector = ball_center - robot_center

        # Normalize vectors
        robot_direction_norm = robot_direction / \
            np.linalg.norm(robot_direction)
        to_ball_vector_norm = to_ball_vector / np.linalg.norm(to_ball_vector)

        # Calculate the angle using the dot product
        dot_product = np.dot(robot_direction_norm, to_ball_vector_norm)
        # Ensure dot product is within valid range
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angle = np.arccos(dot_product)  # This gives the angle in radians

        # Determine the sign of the angle using the cross product
        cross_product = np.cross(robot_direction_norm, to_ball_vector_norm)
        if cross_product < 0:
            angle = -angle  # Clockwise

        # Convert angle to degrees
        angle_degrees = np.degrees(angle)

        return angle_degrees

    def find_blobs(self, color, num_points):
        target_frame = self.morphed_frame if self.morph else self.frame
        contours, _ = self.mask_and_find_contours(target_frame, color=color)
        sorted_contours = self.sort_contours_by_length(
            contours, min_length=10, reverse=True)
        if not sorted_contours:
            return False
        centers = self.find_centers_in_contour_list(
            sorted_contours[:num_points])
        if color == 'orange':
            self.orange_blob_centers = centers
        elif color == 'blue_LED':
            self.blue_LED_centers = centers
        elif color == 'LED':
            self.LED_centers = centers
        return bool(centers)

    def find_robot(self):
        if self.LED_centers:
            self.robot_center = np.mean(self.LED_centers, axis=0)

        # Check if there are enough LED centers and blue LED centers
        if len(self.LED_centers) >= 4 and self.blue_LED_centers:
            # Convert to numpy array for calculations
            led_centers_array = np.array(self.LED_centers)
            blue_led_center = np.array(self.blue_LED_centers[0])

            # Sort the LED centers by distance to the first blue LED center
            sorted_indices = np.argsort(np.linalg.norm(
                led_centers_array - blue_led_center, axis=1))
            sorted_led_centers = led_centers_array[sorted_indices]

            b1, b2, f1, f2 = sorted_led_centers[0], sorted_led_centers[2], sorted_led_centers[1], sorted_led_centers[3]

            # Find the center of the line between the two back LEDs
            back_center = (b1 + b2) / 2

            # Find the center of the line between the two front LEDs
            front_center = (f1 + f2) / 2

            # Get vector from back to front, this is the direction the robot is facing
            direction = front_center - back_center

            # Store the direction for drawing
            self.robot_direction = direction

        else:
            self.robot_direction = None

    def process_frame(self):
        try:
            self.preprocess_frame()
            if self.morph:
                contours, _ = self.mask_and_find_contours(
                    self.frame, color='red')
                sorted_contours = self.sort_contours_by_length(
                    contours, min_length=50, reverse=True)
                if len(sorted_contours) > 2:
                    arena_contour, cross_contour = sorted_contours[1], sorted_contours[2]
                    corners = self.find_sharpest_corners_method1(arena_contour)
                    if corners is not None and len(corners) == 4:
                        corners = np.array([corner.ravel()
                                           for corner in corners], dtype="float32")
                        if not self.four_point_transform(self.frame, corners):
                            print("Skipping frame due to invalid morph points.")
                            return
                        if cross_contour is not None:
                            transformed_contour = cv2.perspectiveTransform(
                                cross_contour.reshape(-1, 1, 2).astype(np.float32), self.M).astype(int)
                            self.fit_rotated_cross_to_contour_method1(
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
                    self.fit_rotated_cross_to_contour_method1(cross_contour)
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
                cv2.circle(self.morphed_frame, center, 5, (0, 255, 0), -1)

            # draw a circle around the first ball in the white ball centers
            cv2.circle(self.morphed_frame,
                       self.white_ball_centers[0], 20, (0, 0, 255), 2)

        if self.blocked_ball_centers:
            for center in self.blocked_ball_centers:
                cv2.circle(self.morphed_frame, center, 5, (0, 0, 0), -1)

        if self.egg_center is not None:
            cv2.circle(self.morphed_frame, self.egg_center, 5, (0, 0, 255), -1)
        if self.cross_lines:
            for line in self.cross_lines:
                cv2.line(self.morphed_frame, line[0], line[1], (255, 0, 0), 2)
        if self.orange_blob_centers:
            for center in self.orange_blob_centers:
                cv2.circle(self.morphed_frame, center, 5, (0, 255, 255), -1)
        if self.LED_centers:
            for center in self.LED_centers:
                cv2.circle(self.morphed_frame, center, 8, (255, 255, 255), -1)
        # if self.blue_LED_centers:
        #     for center in self.blue_LED_centers:
        #         cv2.circle(self.morphed_frame, center, 8, (255, 0, 0), -1)
        if self.robot_center is not None:
            center = tuple(map(int, self.robot_center))
            cv2.circle(self.morphed_frame, center, 8, (0, 0, 5), -1)
            if self.robot_direction is not None:

                end_points = (tuple(map(int, self.robot_center)),
                              tuple(map(int, self.robot_center + 50 * self.robot_direction)))
                cv2.arrowedLine(
                    self.morphed_frame, end_points[0], end_points[1], (0, 0, 0), 2)

        # Draw last valid points on frame
        if self.last_valid_points is not None:
            for pt in self.last_valid_points:
                pt = tuple(map(int, pt))
                cv2.circle(self.frame, pt, 5, (0, 0, 0), -1)

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

    def start_video_stream(self, video_source, morph=True):

        self.morph = morph
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Unable to open video source {video_source}")
            return
        first_valid_points_obtained = False
        while True:
            ret, self.frame = cap.read()

            if not ret:
                print("Error: Unable to read frame from video source")
                break

            self.frame = self.resize_with_aspect_ratio(self.frame, width=640)

            if self.morph and not first_valid_points_obtained:
                for _ in range(10):
                    ret, self.frame = cap.read()

                self.frame = self.resize_with_aspect_ratio(
                    self.frame, width=640)
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

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
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
        self.frame = self.resize_with_aspect_ratio(self.frame, width=640)
        self.preprocess_frame()
        if not ret:
            print("Error: Unable to read frame from video source")
            return
        cv2.namedWindow('Calibration')
        hsv_lower, hsv_upper = self.hsv_ranges[color]
        cv2.createTrackbar('H Lower', 'Calibration',
                           hsv_lower[0], 180, nothing)
        cv2.createTrackbar('S Lower', 'Calibration',
                           hsv_lower[1], 255, nothing)
        cv2.createTrackbar('V Lower', 'Calibration',
                           hsv_lower[2], 255, nothing)
        cv2.createTrackbar('H Upper', 'Calibration',
                           hsv_upper[0], 180, nothing)
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
    video_path = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/camera2/film_2.mp4"
    # video_path = "/dev/video9"

    # camera.calibrate_color('LED', video_path)
    # camera.calibrate_color('red', video_path)
    # camera.calibrate_color('white', video_path)
    # camera.calibrate_color('orange', video_path)
    # camera.calibrate_color('blue_LED', video_path)

    camera.start_video_stream(video_path, morph=True)
