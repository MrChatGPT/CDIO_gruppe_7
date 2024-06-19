import cv2
import numpy as np
from scipy.optimize import minimize

def nothing(x):
    pass

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

    def preprocess_mask(self, mask, kernel_size_open=(5, 5), kernel_size_close=(5, 5)):
        # This function is left empty on purpose. It does not appear to be necessary to process the mask.
        return mask

    def mask_and_find_contours(self, image, color='orange'):
        if color == 'orange':
            hsv_lower, hsv_upper = self.hsv_lower_orange, self.hsv_upper_orange
        else:
            hsv_lower, hsv_upper = self.hsv_lower_white, self.hsv_upper_white
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        processed_mask = self.preprocess_mask(mask)
        contours, hierarchy = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] if hierarchy is not None else []
        return processed_mask, contours, hierarchy

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

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        self.morph_points = tuple(map(tuple, rect))
        maxWidth = int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
        maxHeight = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        self.morphed_image = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return self.morphed_image, M

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

        return self.cross_lines

    def sort_contours_by_length(self, contours):
        return sorted(contours, key=cv2.contourArea, reverse=True)

    def find_white_balls(self):
        if self.morphed_image is None:
            print("Morphed image is not available.")
            return []

        hsv = cv2.cvtColor(self.morphed_image, cv2.COLOR_BGR2HSV)
        white_lower = self.hsv_lower_white
        white_upper = self.hsv_upper_white
        mask = cv2.inRange(hsv, white_lower, white_upper)

        # Use different kernel sizes for opening and closing in white ball preprocessing
        mask = self.preprocess_mask(mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Sort contours by length
        sorted_contours = self.sort_contours_by_length(contours)

        # The largest contour is assumed to be the egg, the rest are white balls
        if sorted_contours:
            egg_contour = sorted_contours[0]
            white_ball_contours = sorted_contours[1:]
        else:
            egg_contour = None
            white_ball_contours = []

        self.white_ball_centers = []
        self.egg_center = None

        # Find and save the egg center
        if egg_contour is not None:
            M = cv2.moments(egg_contour)
            egg_cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
            egg_cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
            self.egg_center = (egg_cx, egg_cy)

        # Calculate average contour length of white balls
        if white_ball_contours:
            avg_contour_length = np.mean([cv2.arcLength(contour, True) for contour in white_ball_contours])

        # Find and save the white ball centers, filtering out small contours
        for contour in white_ball_contours:
            if cv2.arcLength(contour, True) >= 0.5 * avg_contour_length:  # Adjust the factor as needed
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
                cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
                self.white_ball_centers.append((cx, cy))

    def process_frame(self, frame):
        processed_mask, contours, hierarchy = self.mask_and_find_contours(frame, color='orange')

        # Sort contours by length
        sorted_contours = self.sort_contours_by_length(contours)

        if len(sorted_contours) > 1:
            arena_contour = sorted_contours[1]
            arena_contour = cv2.approxPolyDP(arena_contour, 0.01 * cv2.arcLength(arena_contour, True), True)
            cross_contour = sorted_contours[2]

            if arena_contour is not None:
                corners = self.find_sharpest_corners(processed_mask, arena_contour, num_corners=4)

                if corners is not None and len(corners) == 4:
                    corners = np.array([corner.ravel() for corner in corners], dtype="float32")
                    warped_image, M = self.four_point_transform(frame, corners)

                    if cross_contour is not None:
                        cross_contour_points = np.array(cross_contour, dtype='float32')
                        transformed_contour = cv2.perspectiveTransform(cross_contour_points.reshape(-1, 1, 2), M)

                        # Fit a rotated cross to the transformed longest child contour
                        self.fit_rotated_cross_to_contour(transformed_contour.astype(int))

                        # Draw the cross lines on the image
                        for line in self.cross_lines:
                            cv2.line(warped_image, line[0], line[1], (0, 0, 255), 2)

                        # Find white balls in the morphed image
                        self.find_white_balls()

                        # Draw the white balls and the egg center on the image
                        for center in self.white_ball_centers:
                            cv2.circle(warped_image, center, 3, (0, 255, 0), -1)
                        
                        if self.egg_center is not None:
                            cv2.circle(warped_image, self.egg_center, 3, (0, 0, 255), -1)

                        return warped_image

                    else:
                        print("The first child contour has no children.")
                else:
                    print("Could not find four corners in the first child contour.")
            else:
                print("The top contour has no children.")
        else:
            print("Not enough contours found.")

        return frame

    def start_video_stream(self, video_source):
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print(f"Error: Unable to open video source {video_source}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to read frame from video source")
                break

            processed_frame = self.process_frame(frame)

            # Display the original frame with contours and the processed frame separately
            cv2.imshow('Processed Frame', processed_frame)

            # Step through frames by pressing any key
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

        # Create a window
        cv2.namedWindow('Calibration')

        # Initialize trackbars
        if color == 'white':
            h_lower, s_lower, v_lower = self.hsv_lower_white
            h_upper, s_upper, v_upper = self.hsv_upper_white
        else:
            h_lower, s_lower, v_lower = self.hsv_lower_orange
            h_upper, s_upper, v_upper = self.hsv_upper_orange

        cv2.createTrackbar('H Lower', 'Calibration', h_lower, 180, nothing)
        cv2.createTrackbar('S Lower', 'Calibration', s_lower, 255, nothing)
        cv2.createTrackbar('V Lower', 'Calibration', v_lower, 255, nothing)
        cv2.createTrackbar('H Upper', 'Calibration', h_upper, 180, nothing)
        cv2.createTrackbar('S Upper', 'Calibration', s_upper, 255, nothing)
        cv2.createTrackbar('V Upper', 'Calibration', v_upper, 255, nothing)

        while True:
            # Get the current positions of the trackbars
            h_lower = cv2.getTrackbarPos('H Lower', 'Calibration')
            s_lower = cv2.getTrackbarPos('S Lower', 'Calibration')
            v_lower = cv2.getTrackbarPos('V Lower', 'Calibration')
            h_upper = cv2.getTrackbarPos('H Upper', 'Calibration')
            s_upper = cv2.getTrackbarPos('S Upper', 'Calibration')
            v_upper = cv2.getTrackbarPos('V Upper', 'Calibration')

            # Define the HSV range based on the trackbar positions
            lower_hsv = np.array([h_lower, s_lower, v_lower])
            upper_hsv = np.array([h_upper, s_upper, v_upper])

            # Convert the frame to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Create a mask based on the HSV range
            mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

            # Find contours
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the contours on the original frame
            contour_image = frame.copy()
            cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

            # Display the original frame, the binary mask, and the contour image
            cv2.imshow('Original Frame', frame)
            cv2.imshow('Binary Mask', mask)
            cv2.imshow('Contours', contour_image)

            # Press 's' to save the current settings
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if color == 'white':
                    self.hsv_lower_white = lower_hsv
                    self.hsv_upper_white = upper_hsv
                else:
                    self.hsv_lower_orange = lower_hsv
                    self.hsv_upper_orange = upper_hsv
                break

            # Press 'q' to quit without saving
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
