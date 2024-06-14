import cv2
import numpy as np
from scipy.optimize import minimize

class Camera2:
    def __init__(self):
        # Initialize with default HSV thresholds
        self.hsv_lower_orange = np.array([0, 130, 196])
        self.hsv_upper_orange = np.array([9, 255, 255])
        self.hsv_lower_white = np.array([0, 0, 233])
        self.hsv_upper_white = np.array([180, 58, 255])
        
        self.morph_points = None
        self.morphed_image = None
        self.cross_lines = None
        self.white_ball_centers = []
        self.egg_center = None
        self.last_cross_angle = 0  # Initialize the last found rotation angle
        self.M = None  # Transformation matrix
        self.last_valid_points = None  # Store last valid points

    def preprocess_mask(self, mask):
        return mask

    def mask_and_find_contours(self, image, color='orange'):
        if color == 'orange':
            hsv_lower, hsv_upper = self.hsv_lower_orange, self.hsv_upper_orange
        else:
            hsv_lower, hsv_upper = self.hsv_lower_white, self.hsv_upper_white
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        processed_mask = self.preprocess_mask(mask)
        contours, _ = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours, processed_mask

    def find_sharpest_corners(self, mask, contour, num_corners=4):
        contour_mask = np.zeros_like(mask)
        cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
        corners = cv2.goodFeaturesToTrack(contour_mask, maxCorners=num_corners, qualityLevel=0.01, minDistance=10)
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
                print(f"Rejected morph points due to excessive deviation: {deviation:.2f}%")
                # Use the last valid transformation matrix if available
                if self.last_valid_M is not None:
                    self.M = self.last_valid_M
                    self.morphed_image = cv2.warpPerspective(image, self.M, (image.shape[1], image.shape[0]))
                return False

        self.last_valid_points = rect
        self.morph_points = tuple(map(tuple, rect))
        maxWidth = int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
        maxHeight = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        self.M = cv2.getPerspectiveTransform(rect, dst)
        self.last_valid_M = self.M  # Update the last valid transformation matrix
        self.morphed_image = cv2.warpPerspective(image, self.M, (maxWidth, maxHeight))
        return True

    def distance_to_cross(self, points, cx, cy, angle):
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        total_distance = np.sum(np.minimum(np.abs(cos_angle * (points[:, 0] - cx) + sin_angle * (points[:, 1] - cy)),
                                           np.abs(-sin_angle * (points[:, 0] - cx) + cos_angle * (points[:, 1] - cy)))**2)
        return total_distance

    def fit_rotated_cross_to_contour(self, contour):
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0

        contour_points = contour.reshape(-1, 2)
        max_distance = np.max(np.linalg.norm(contour_points - np.array([cx, cy]), axis=1))

        # Use the last found rotation angle as the initial guess for optimization
        result = minimize(lambda angle: self.distance_to_cross(contour_points, cx, cy, angle), self.last_cross_angle)
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

    def sort_contours_by_length(self, contours):
        min_area = 10  # Set a minimum area threshold
        return [contour for contour in sorted(contours, key=cv2.contourArea, reverse=True) if cv2.contourArea(contour) >= min_area]

    def find_white_balls(self):
        if self.morphed_image is None:
            print("Morphed image is not available.")
            return

        hsv = cv2.cvtColor(self.morphed_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower_white, self.hsv_upper_white)
        mask = self.preprocess_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        sorted_contours = self.sort_contours_by_length(contours)

        if sorted_contours:
            egg_contour = sorted_contours[0]
            white_ball_contours = sorted_contours[1:]
        else:
            egg_contour = None
            white_ball_contours = []

        self.white_ball_centers = []
        self.egg_center = None

        if egg_contour is not None:
            M = cv2.moments(egg_contour)
            egg_cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            egg_cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
            self.egg_center = (egg_cx, egg_cy)

        if white_ball_contours:
            avg_contour_length = np.mean([cv2.arcLength(contour, True) for contour in white_ball_contours])

        for contour in white_ball_contours:
            if cv2.arcLength(contour, True) >= 0.5 * avg_contour_length:  # Adjust the factor as needed
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
                cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
                self.white_ball_centers.append((cx, cy))

    def process_frame(self, frame):
        try:
            contours, processed_mask = self.mask_and_find_contours(frame, color='orange')
            sorted_contours = self.sort_contours_by_length(contours)

            if len(sorted_contours) > 1:
                arena_contour = sorted_contours[1]
                arena_contour = cv2.approxPolyDP(arena_contour, 0.01 * cv2.arcLength(arena_contour, True), True)
                cross_contour = sorted_contours[2]

                if arena_contour is not None:
                    corners = self.find_sharpest_corners(processed_mask, arena_contour, num_corners=4)

                    if corners is not None and len(corners) == 4:
                        corners = np.array([corner.ravel() for corner in corners], dtype="float32")
                        if not self.four_point_transform(frame, corners):
                            print("Skipping frame due to invalid morph points.")
                            return

                        if cross_contour is not None:
                            cross_contour_points = np.array(cross_contour, dtype='float32')
                            transformed_contour = cv2.perspectiveTransform(cross_contour_points.reshape(-1, 1, 2), self.M)

                            self.fit_rotated_cross_to_contour(transformed_contour.astype(int))

                            for line in self.cross_lines:
                                cv2.line(self.morphed_image, line[0], line[1], (0, 0, 255), 2)

                            self.find_white_balls()

                            for center in self.white_ball_centers:
                                cv2.circle(self.morphed_image, center, 3, (0, 255, 0), -1)
                            
                            if self.egg_center is not None:
                                cv2.circle(self.morphed_image, self.egg_center, 3, (0, 0, 255), -1)
            else:
                self.morphed_image = frame
        except IndexError as e:
            print(f"IndexError: {e}")
            self.morphed_image = frame
        except Exception as e:
            print(f"Error: {e}")
            self.morphed_image = frame

    def start_video_stream(self, video_source):
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"Error: Unable to open video source {video_source}")
            return

        first_valid_points_obtained = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from video source")
                break

            frame = cv2.resize(frame, (640, 480))

            if not first_valid_points_obtained:
                self.process_frame(frame)
                if self.last_valid_points is not None:
                    print("First set of valid points obtained.")
                    first_valid_points_obtained = True

                # Display the initial frame with detected points
                for pt in self.last_valid_points:
                    pt = tuple(map(int, pt))  # Ensure pt is a tuple of integers
                    cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                cv2.imshow('Initial Frame', frame)

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
                self.process_frame(frame)
                cv2.imshow('Processed Frame', self.morphed_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

    def calibrate_color(self, color):
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Unable to open video source {video_path}")
            return

        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame from video source")
            return

        frame = cv2.resize(frame, (640, 480))

        cv2.namedWindow('Calibration')

        if color == 'white':
            h_lower, s_lower, v_lower = self.hsv_lower_white
            h_upper, s_upper, v_upper = self.hsv_upper_white
        else:
            h_lower, s_lower, v_lower = self.hsv_lower_orange
            h_upper, s_upper, v_upper = self.hsv_upper_orange

        cv2.createTrackbar('H Lower', 'Calibration', h_lower, 180, )
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

            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contour_image = frame.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

            cv2.imshow('Original Frame', frame)
            cv2.imshow('Binary Mask', mask)
            cv2.imshow('Contours', contour_image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if color == 'white':
                    self.hsv_lower_white = lower_hsv
                    self.hsv_upper_white = upper_hsv
                else:
                    self.hsv_lower_orange = lower_hsv
                    self.hsv_upper_orange = upper_hsv
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
    camera.calibrate_color('white')

    # Uncomment the line below to calibrate the orange color
    # camera.calibrate_color('orange')

    camera.start_video_stream(video_path)

