import cv2
import numpy as np
from scipy.optimize import minimize


class Camera2:
    def __init__(self, hsv_thresholds):
        self.hsv_lower, self.hsv_upper = hsv_thresholds
        self.morph_points = None
        self.morphed_image = None
        self.cross_lines = None
        self.white_ball_centers = []
        self.egg_center = None
        self.last_cross_angle = 0  # Initialize the last found rotation angle

    def preprocess_mask(self, mask, kernel_size_open=(5, 5), kernel_size_close=(5, 5)):
        kernel_open = np.ones(kernel_size_open, np.uint8)
        kernel_close = np.ones(kernel_size_close, np.uint8)

        # Opening to remove small noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        # Closing to close small gaps
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        return mask

    def mask_and_find_contours(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        processed_mask = self.preprocess_mask(mask)
        contours, hierarchy = cv2.findContours(
            processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hierarchy = hierarchy[0] if hierarchy is not None else []
        return processed_mask, contours, hierarchy

    def find_sharpest_corners(self, mask, contour, num_corners=4):
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

    def four_point_transform(self, image, pts):
        rect = self.order_points(pts)
        self.morph_points = tuple(map(tuple, rect))
        maxWidth = int(
            max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0])))
        maxHeight = int(
            max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))
        dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1,
                       maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        self.morphed_image = cv2.warpPerspective(
            image, M, (maxWidth, maxHeight))
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
        max_distance = np.max(np.linalg.norm(
            contour_points - np.array([cx, cy]), axis=1))

        # Use the last found rotation angle as the initial guess for optimization
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

        return self.cross_lines

    def sort_contours_by_length(self, contours):
        return sorted(contours, key=cv2.contourArea, reverse=True)

    def find_white_balls(self):
        if self.morphed_image is None:
            print("Morphed image is not available.")
            return []

        hsv = cv2.cvtColor(self.morphed_image, cv2.COLOR_BGR2HSV)
        white_lower = np.array([0, 0, 245])
        white_upper = np.array([180, 54, 255])
        mask = cv2.inRange(hsv, white_lower, white_upper)

        # Use different kernel sizes for opening and closing in white ball preprocessing
        mask = self.preprocess_mask(
            mask, kernel_size_open=(5, 5), kernel_size_close=(5, 5))

        # Additional erosion to separate touching blobs
        kernel_erode = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel_erode, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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
            avg_contour_length = np.mean(
                [cv2.arcLength(contour, True) for contour in white_ball_contours])

        # Find and save the white ball centers, filtering out small contours
        for contour in white_ball_contours:
            # Adjust the factor as needed
            if cv2.arcLength(contour, True) >= 0.5 * avg_contour_length:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else 0
                cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else 0
                self.white_ball_centers.append((cx, cy))

    def start(self, image):
        processed_mask, contours, hierarchy = self.mask_and_find_contours(
            image)

        # Sort contours by length
        sorted_contours = self.sort_contours_by_length(contours)

        if len(sorted_contours) > 1:
            arena_contour = sorted_contours[1]
            arena_contour = cv2.approxPolyDP(
                arena_contour, 0.01 * cv2.arcLength(arena_contour, True), True)
            cross_contour = sorted_contours[2]

            if arena_contour is not None:
                print("Found the top contour.")

                corners = self.find_sharpest_corners(
                    processed_mask, arena_contour, num_corners=4)

                if corners is not None and len(corners) == 4:
                    corners = np.array([corner.ravel()
                                       for corner in corners], dtype="float32")
                    warped_image, M = self.four_point_transform(image, corners)

                    if cross_contour is not None:
                        cross_contour_points = np.array(
                            cross_contour, dtype='float32')
                        transformed_contour = cv2.perspectiveTransform(
                            cross_contour_points.reshape(-1, 1, 2), M)

                        # Fit a rotated cross to the transformed longest child contour
                        self.fit_rotated_cross_to_contour(
                            transformed_contour.astype(int))

                        # Draw the cross lines on the image
                        for line in self.cross_lines:
                            cv2.line(warped_image,
                                     line[0], line[1], (0, 0, 255), 2)

                        # Find white balls in the morphed image
                        self.find_white_balls()

                        # Draw the white balls and the egg center on the image
                        for center in self.white_ball_centers:
                            cv2.circle(warped_image, center,
                                       3, (0, 255, 0), -1)

                        if self.egg_center is not None:
                            cv2.circle(warped_image, self.egg_center,
                                       3, (0, 0, 255), -1)

                        return warped_image

                    else:
                        print("The first child contour has no children.")
                else:
                    print("Could not find four corners in the first child contour.")
            else:
                print("The top contour has no children.")
        else:
            print("Not enough contours found.")

        return None


# Main part to test the Camera2 class
if __name__ == "__main__":
    hsv_thresholds = (np.array([0, 99, 201]), np.array([180, 255, 255]))
    camera = Camera2(hsv_thresholds)

    imagePath = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/camera2/testImg1.jpg"
    image = cv2.imread(imagePath)

    if image is None:
        print(f"Error: Unable to load image from {imagePath}")
    else:
        result_image = camera.start(image)

        if result_image is not None:
            cv2.imshow(
                'Warped Image with Transformed Contour and Rotated Cross', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        else:
            print("Image processing failed.")
