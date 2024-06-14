import cv2
import numpy as np
from scipy.optimize import minimize

def preprocess_mask(mask):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

def mask_and_find_contours(image, hsv_lower, hsv_upper):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    processed_mask = preprocess_mask(mask)
    contours, hierarchy = cv2.findContours(processed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return processed_mask, contours, hierarchy[0] if hierarchy is not None else []

def find_longest_child(contours, hierarchy, parent_index):
    if parent_index == -1:
        return -1

    child_index = hierarchy[parent_index][2]
    if child_index == -1:
        return -1

    longest_child_index, max_length = child_index, cv2.arcLength(contours[child_index], True)

    next_sibling = hierarchy[child_index][0]
    while next_sibling != -1:
        length = cv2.arcLength(contours[next_sibling], True)
        if length > max_length:
            longest_child_index, max_length = next_sibling, length
        next_sibling = hierarchy[next_sibling][0]

    return longest_child_index

def find_sharpest_corners(mask, contour, num_corners=4):
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [contour], -1, 255, thickness=cv2.FILLED)
    corners = cv2.goodFeaturesToTrack(contour_mask, maxCorners=num_corners, qualityLevel=0.01, minDistance=10)
    return corners.astype(int) if corners is not None else None

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s, diff = pts.sum(axis=1), np.diff(pts, axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    maxWidth, maxHeight = int(max(np.linalg.norm(rect[2] - rect[3]), np.linalg.norm(rect[1] - rect[0]))), int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[0] - rect[3])))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight)), M

def distance_to_cross(points, cx, cy, angle):
    cos_angle, sin_angle = np.cos(angle), np.sin(angle)
    total_distance = sum(min(abs(cos_angle * (x - cx) + sin_angle * (y - cy)), abs(-sin_angle * (x - cx) + cos_angle * (y - cy)))**2 for x, y in points)
    return total_distance

def fit_rotated_cross_to_contour(contour):
    M = cv2.moments(contour)
    cx, cy = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])) if M['m00'] != 0 else (0, 0)

    contour_points = contour.reshape(-1, 2)
    max_distance = max(np.linalg.norm(contour_points - np.array([cx, cy]), axis=1))

    result = minimize(lambda angle: distance_to_cross(contour_points, cx, cy, angle), 0)
    best_angle, cross_length = result.x[0], max_distance

    cos_angle, sin_angle = np.cos(best_angle), np.sin(best_angle)
    endpoints = [
        ((int(cx + cross_length * cos_angle), int(cy + cross_length * sin_angle)), 
         (int(cx - cross_length * cos_angle), int(cy - cross_length * sin_angle))),
        ((int(cx + cross_length * -sin_angle), int(cy + cross_length * cos_angle)), 
         (int(cx - cross_length * -sin_angle), int(cy - cross_length * cos_angle)))
    ]

    return endpoints


def sort_contours_by_length(contours):
    return sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)

# Example usage:
imagePath = "/home/madsr2d2/sem4/CDIO/CDIO_gruppe_7/camera2/testImg1.jpg"
image = cv2.imread(imagePath)

if image is None:
    print(f"Error: Unable to load image from {imagePath}")
else:
    hsv_lower, hsv_upper = np.array([0, 130, 196]), np.array([9, 255, 255])
    processed_mask, contours, hierarchy = mask_and_find_contours(image, hsv_lower, hsv_upper)

    # Sort contours by length
    sorted_contours = sort_contours_by_length(contours)

    # Draw all contours on the original image
    original_image_with_contours = image.copy()

    arena_contour = sorted_contours[1]

    cv2.drawContours(original_image_with_contours, arena_contour, -1, (0, 255, 0), 2)
    cv2.imshow('Original Image with All Contours', original_image_with_contours)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Use the sorted contours for further processing

    if arena_contour != None:
        
        corners = find_sharpest_corners(processed_mask, arena_contour, num_corners=4)

        if corners is not None and len(corners) == 4:
            corners = np.array([corner.ravel() for corner in corners], dtype="float32")
            warped_image, M = four_point_transform(image, corners)

            cross_contour = sorted_contours[2]
            if cross_contour != None:

                cross_contour_points = np.array(cross_contour, dtype='float32')
                transformed_contour = cv2.perspectiveTransform(cross_contour_points.reshape(-1, 1, 2), M)

                # Fit a rotated cross to the transformed longest child contour
                cross_lines = fit_rotated_cross_to_contour(transformed_contour.astype(int))

                # Draw the cross lines on the image
                for line in cross_lines:
                    cv2.line(warped_image, line[0], line[1], (0, 255, 0), 2)

                cv2.imshow('Warped Image with Transformed Contour and Rotated Cross', warped_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("The first child contour has no children.")
        else:
            print("Could not find four corners in the first child contour.")
    else:
        print("The top contour has no children.")

