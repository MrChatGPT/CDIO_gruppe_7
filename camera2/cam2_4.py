import cv2
import numpy as np
from scipy.optimize import minimize


class Camera2:
    def __init__(self):
        # Initialize with default HSV thresholds
        self.hsv_lower_red = np.array([0, 130, 196])
        self.hsv_upper_red = np.array([9, 255, 255])
        self.hsv_lower_white = np.array([0, 0, 233])
        self.hsv_upper_white = np.array([180, 58, 255])

        self.morph_points = None
        self.morphed_frame = None
        self.frame = None
        self.cross_lines = None
        self.white_ball_centers = []
        self.egg_center = None
        self.last_cross_angle = 0  # Initialize the last found rotation angle
        self.M = None  # Transformation matrix
        self.last_valid_points = None  # Store last valid points

    def preprocess_mask(self, mask):
        # Placeholder for future preprocessing steps, if needed
        return mask

    def mask_and_find_contours(self, image, color):
        if color == 'red':
            hsv_lower, hsv_upper = self.hsv_lower_red, self.hsv_upper_red
        elif color == 'white':
            hsv_lower, hsv_upper = self.hsv_lower_white, self.hsv_upper_white

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        processed_mask = self.preprocess_mask(mask)
        contours, _ = cv2.findContours(
            processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, processed_mask

    def find_sharpest_corners_method1(self, contour, num_corners=4, epsilon_factor=0.02):

        # Approximate the contour with a simpler polygon
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx_corners = cv2.approxPolyDP(contour, epsilon, True)

        # If the approximation has more than the desired number of corners, reduce the epsilon factor
        while len(approx_corners) > num_corners:
            epsilon_factor += 0.01
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx_corners = cv2.approxPolyDP(contour, epsilon, True)

        # If we have exactly the number of desired corners, return them
        if len(approx_corners) == num_corners:
            return approx_corners.reshape(-1, 2).astype(int)
        else:
            return None

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
            if deviation > 10:  # Set a threshold percentage deviation
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
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        total_distance = np.sum(np.minimum(
            np.abs(cos_angle * (points[:, 0] - cx) +
                   sin_angle * (points[:, 1] - cy)),
            np.abs(-sin_angle * (points[:, 0] - cx) +
                   cos_angle * (points[:, 1] - cy))
        )**2)
        return total_distance

    def fit_rotated_cross_to_contour_method1(self, contour):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        self.cross_lines = [
            (tuple(box[0]), tuple(box[2])),
            (tuple(box[1]), tuple(box[3]))
        ]

    def fit_rotated_cross_to_contour_method2(self, contour):
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

        contour_points = contour.reshape(-1, 2)
        max_distance = np.max(np.linalg.norm(
            contour_points - np.array([cx, cy]), axis=1))

        result = minimize(lambda angle: self.distance_to_cross(
            contour_points, cx, cy, angle), self.last_cross_angle)
        best_angle = result.x[0]
        self.last_cross_angle = best_angle  # Update the last found rotation angle
        cross_length = max_distance

        cos_angle = np.cos(best_angle)
        sin_angle = np.sin(best_angle)
        self.cross_lines = [
            ((int(cx + cross_length * cos_angle), int(cy + cross_length * sin_angle)),
             (int(cx - cross_length * cos_angle), int(cy - cross_length * sin_angle))),
            ((int(cx + cross_length * -sin_angle), int(cy + cross_length * cos_angle)),
             (int(cx - cross_length * -sin_angle), int(cy - cross_length * cos_angle)))
        ]

    def sort_contours_by_length(self, contours, min_length=10, reverse=True):
        return [contour for contour in sorted(contours, key=cv2.contourArea, reverse=reverse) if cv2.arcLength(contour, True) >= min_length]

    def find_centers_in_contour_list(self, contours):
        centers = []
        for contour in contours:
            M = cv2.moments(contour)
            cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
            centers.append((cx, cy))
        return centers

    def find_white_blobs(self):
        if self.morphed_frame is None:
            print("Morphed image is not available.")
            return False

        contours, _ = self.mask_and_find_contours(
            self.morphed_frame, color='white')

        if not contours:
            return False

        sorted_contours = self.sort_contours_by_length(
            contours, min_length=10, reverse=True)

        if len(sorted_contours) == 1:
            egg_contour = contours[0]
            white_ball_contours = []
            self.egg_center = self.find_centers_in_contour_list([egg_contour])[
                0]
            self.white_ball_centers = []
            return True
        else:
            egg_contour = sorted_contours[0]
            self.egg_center = self.find_centers_in_contour_list([egg_contour])[
                0]
            white_ball_contours = sorted_contours[1:]
            avg_white_ball_contour_length = np.mean(
                [cv2.arcLength(contour, True) for contour in white_ball_contours])
            white_ball_contours = [contour for contour in white_ball_contours if cv2.arcLength(
                contour, True) >= 0.5 * avg_white_ball_contour_length]
            self.white_ball_centers = self.find_centers_in_contour_list(
                white_ball_contours)
            return True

    def process_frame(self):
        try:
            # Mask and find contours
            contours, _ = self.mask_and_find_contours(
                self.frame, color='red')
            sorted_contours = self.sort_contours_by_length(
                contours, min_length=200, reverse=True)

            if len(sorted_contours) > 2:
                # Get arena and cross contours
                arena_contour = sorted_contours[1]
                cross_contour = sorted_contours[2]

                # Find sharpest corners
                corners = self.find_sharpest_corners_method1(
                    arena_contour)  # Method 1 is faster

                if corners is not None and len(corners) == 4:
                    corners = np.array([corner.ravel()
                                       for corner in corners], dtype="float32")

                    # Perform perspective transform
                    if not self.four_point_transform(self.frame, corners):
                        print("Skipping frame due to invalid morph points.")
                        return

                    # Process cross contour in the transformed image
                    if cross_contour is not None:
                        cross_contour_points = np.array(
                            cross_contour, dtype='float32')
                        transformed_contour = cv2.perspectiveTransform(
                            cross_contour_points.reshape(-1, 1, 2), self.M)
                        self.fit_rotated_cross_to_contour_method1(
                            transformed_contour.astype(int))

                    # Find and draw white blobs
                    self.find_white_blobs()

                    # Draw on the morphed frame
                    if self.white_ball_centers:
                        for center in self.white_ball_centers:
                            cv2.circle(self.morphed_frame,
                                       center, 3, (0, 255, 0), -1)

                    if self.egg_center is not None:
                        cv2.circle(self.morphed_frame,
                                   self.egg_center, 3, (0, 0, 255), -1)

                    if self.cross_lines:
                        for line in self.cross_lines:
                            cv2.line(self.morphed_frame,
                                     line[0], line[1], (255, 0, 0), 2)
            else:
                self.morphed_frame = self.frame
        except IndexError as e:
            self.morphed_frame = self.frame

    def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image

        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))

        return cv2.resize(image, dim, interpolation=inter)

    def start_video_stream(self, video_source):
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

            if not first_valid_points_obtained:
                self.process_frame()
                if self.last_valid_points is not None:
                    print("First set of valid points obtained.")
                    first_valid_points_obtained = True

                for pt in self.last_valid_points:
                    pt = tuple(map(int, pt))
                    cv2.circle(self.frame, pt, 5, (0, 255, 0), -1)
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

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def calibrate_color(self, color):
        def nothing(x):
            pass

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video source {video_path}")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from video source")
            return

        frame = self.resize_with_aspect_ratio(frame, width=640)

        cv2.namedWindow('Calibration')

        if color == 'white':
            h_lower, s_lower, v_lower = self.hsv_lower_white
            h_upper, s_upper, v_upper = self.hsv_upper_white
        elif color == 'red':
            h_lower, s_lower, v_lower = self.hsv_lower_red
            h_upper, s_upper, v_upper = self.hsv_upper_red

        cv2.createTrackbar('H Lower', 'Calibration', h_lower, 180, nothing)
        cv2.createTrackbar('S Lower', 'Calibration', s_lower, 255, nothing)
        cv2.createTrackbar('V Lower', 'Calibration', v_lower, 255, nothing)
        cv2.createTrackbar('H Upper', 'Calibration', h_upper, 180, nothing)
        cv2.createTrackbar('S Upper', 'Calibration', s_upper, 255, nothing)
        cv2.createTrackbar('V Upper', 'Calibration', v_upper, 255, nothing)

        while True:
            h_lower = cv2.getTrackbarPos('H Lower', 'Calibration')
            s_lower = cv2.getTrackbarPos('S Lower', 'Calibration')
            v_lower = cv2.getTrackbarPos('V Lower', 'Calibration')
            h_upper = cv2.getTrackbarPos('H Upper', 'Calibration')
            s_upper = cv2.getTrackbarPos('S Upper', 'Calibration')
            v_upper = cv2.getTrackbarPos('V Upper', 'Calibration')

            lower_hsv = np.array([h_lower, s_lower, v_lower])
            upper_hsv = np.array([h_upper, s_upper, v_upper])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = frame.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

            cv2.imshow('Original Frame', frame)
            cv2.imshow('Binary Mask', mask)
            cv2.imshow('Contours', contour_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                if color == 'white':
                    self.hsv_lower_white = lower_hsv
                    self.hsv_upper_white = upper_hsv
                else:
                    self.hsv_lower_red = lower_hsv
                    self.hsv_upper_red = upper_hsv
                break
            elif key == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


# Main part to test the Camera2 class with video stream and calibration
if __name__ == "__main__":
    camera = Camera2()
    video_path = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/camera2/seme.mp4"

    # Uncomment the line below to calibrate the white color
    # camera.calibrate_color('white')

    # Uncomment the line below to calibrate the red color
    # camera.calibrate_color('red')

    camera.start_video_stream(video_path)
