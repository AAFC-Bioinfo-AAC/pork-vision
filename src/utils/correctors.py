# correctors.py

import cv2
import numpy as np
from skimage import draw, measure
from .lines import find_nearest_contour_point

def distance_corrector(
    img,
    orientation,
    rotated_fat_box,
    rotated_muscle_box,
    rotated_muscle_mask,
    point_where_muscle_connects_to_fat,
    angle,
    center,
):
    """
    Corrects the distance between muscle and fat layers based on the orientation and angle.

    Parameters:
    img (numpy.ndarray): The input image.
    orientation (str): Orientation of the fat layer relative to the muscle layer.
    rotated_fat_box (numpy.ndarray): Bounding box for the rotated fat layer.
    rotated_muscle_box (numpy.ndarray): Bounding box for the rotated muscle layer.
    rotated_muscle_mask (numpy.ndarray): Binary mask of the rotated muscle layer.
    point_where_muscle_connects_to_fat (numpy.ndarray): Point where the muscle connects to the fat.
    angle (float): Rotation angle in degrees.
    center (tuple): The center point (x, y) around which the image will be rotated.

    Returns:
    tuple: A tuple containing:
        - new_p2 (numpy.ndarray): Corrected point where the muscle connects to the fat.
        - muscle_mid (numpy.ndarray): Midpoint of the muscle bounding box.
    """
    fat_bbox = rotated_fat_box
    muscle_bbox = rotated_muscle_box
    p2 = point_where_muscle_connects_to_fat
    rotated_contours, _ = cv2.findContours(
        rotated_muscle_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    rotated_contours = rotated_contours[0].reshape(-1, 2)

    if orientation == "FAT_TOP":
        new_p2, muscle_mid, fat_pt = rotation_detector_by_angle_corrector(
            "FAT_TOP", angle, fat_bbox, muscle_bbox, p2
        )
        new_p2 = find_nearest_contour_point(rotated_contours, new_p2)

    elif orientation == "FAT_BOTTOM":
        new_p2, muscle_mid, fat_pt = rotation_detector_by_angle_corrector(
            "FAT_BOTTOM", angle, fat_bbox, muscle_bbox, p2
        )
        new_p2 = find_nearest_contour_point(rotated_contours, new_p2)

    elif orientation == "FAT_RIGHT":
        new_p2, muscle_mid, fat_pt = rotation_detector_by_angle_corrector(
            "FAT_RIGHT", angle, fat_bbox, muscle_bbox, p2
        )
        new_p2 = find_nearest_contour_point(rotated_contours, new_p2)

    elif orientation == "FAT_LEFT":
        new_p2, muscle_mid, fat_pt = rotation_detector_by_angle_corrector(
            "FAT_LEFT", angle, fat_bbox, muscle_bbox, p2
        )
        new_p2 = find_nearest_contour_point(rotated_contours, new_p2)

    return new_p2, muscle_mid


def line_to_fat_corrector(new_p1, endpt_x, endpt_y, rotated_fat_mask):
    """
    Corrects the distance between muscle and fat using the push-pull method.

    Parameters:
    new_p1 (numpy.ndarray): Starting point of the line segment.
    endpt_x (int): X-coordinate of the extended endpoint.
    endpt_y (int): Y-coordinate of the extended endpoint.
    rotated_fat_mask (numpy.ndarray): Binary mask of the rotated fat layer.

    Returns:
    numpy.ndarray: The point on the fat layer where the line intersects.
    """
    discrete_line = list(
        zip(*draw.line(*new_p1.astype("int32"), endpt_x, endpt_y))
    )
    contours, _ = cv2.findContours(
        rotated_fat_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    pts = measure.points_in_poly(discrete_line, contours[0].reshape(-1, 2))
    disc_line_fat = np.array(discrete_line)[pts]
    return disc_line_fat[-1]

def rotation_detector_by_angle_corrector(
    current_fat_placement, angle, fat_bbox, muscle_bbox, p2
):
    """
    Corrects the position of a point based on the angle of rotation and the placement of the fat layer.

    Parameters:
    current_fat_placement (str): The current placement of the fat layer ("FAT_TOP", "FAT_BOTTOM", "FAT_LEFT", "FAT_RIGHT").
    angle (float): The angle of rotation calculated by the ellipsoid method.
    fat_bbox (numpy.ndarray): Bounding box for the fat.
    muscle_bbox (numpy.ndarray): Bounding box for the muscle.
    p2 (numpy.ndarray): The point to be corrected.

    Returns:
    tuple: A tuple containing:
        - p2 (numpy.ndarray): The corrected point.
        - muscle_mid (numpy.ndarray): The midpoint of the muscle bounding box.
        - fat_pt (numpy.ndarray): The midpoint of the fat bounding box.
    """
    max_distance = 6.5
    min_distance = 5.5
    common_distance = 6.0

    angle_adj = abs(90 - angle) if angle > 0 else 90 + angle

    if angle_adj < 45 and current_fat_placement == "FAT_BOTTOM":
        fat_pt = (fat_bbox[1] + fat_bbox[2]) / 2
        muscle_mid = (muscle_bbox[0] + muscle_bbox[1]) / 2

        if (abs(p2 - fat_pt)[0] / 140 > max_distance):
            while (abs(p2 - fat_pt)[0] / 140) > common_distance:
                p2[0] += 1
                p2[1] -= 1
        elif (abs(p2 - fat_pt)[0] / 140 < min_distance):
            while (abs(p2 - fat_pt)[0] / 140) < common_distance:
                p2[0] -= 1
                p2[1] -= 1

    elif angle_adj > 45 and current_fat_placement == "FAT_BOTTOM":
        fat_pt = (fat_bbox[1] + fat_bbox[2]) / 2
        muscle_mid = (muscle_bbox[0] + muscle_bbox[1]) / 2

        if (abs(p2 - fat_pt)[1] / 140 > max_distance):
            while (abs(p2 - fat_pt)[1] / 140) > common_distance:
                p2[0] -= p2[0] - 1
                p2[1] -= p2[1] - 1
        elif (abs(p2 - fat_pt)[1] / 140 < min_distance):
            while (abs(p2 - fat_pt)[1] / 140) < common_distance:
                p2[0] -=  1
                p2[1] += 1

    elif angle_adj < 45 and current_fat_placement == "FAT_TOP":
        fat_pt = (fat_bbox[0] + fat_bbox[3]) / 2
        muscle_mid = (muscle_bbox[2] + muscle_bbox[3]) / 2

        if (abs(p2 - fat_pt)[0] / 140 > max_distance):
            while (abs(p2 - fat_pt)[0] / 140) > common_distance:
                p2[0] -= 1
                p2[1] += 1
        elif (abs(p2 - fat_pt)[0] / 140 < min_distance):
            while (abs(p2 - fat_pt)[0] / 140) < common_distance:
                p2[0] += 1
                p2[1] += 1

    elif angle_adj > 45 and current_fat_placement == "FAT_TOP":
        fat_pt = (fat_bbox[0] + fat_bbox[3]) / 2
        muscle_mid = (muscle_bbox[2] + muscle_bbox[3]) / 2

        if (abs(p2 - fat_pt)[1] / 140 > max_distance):
            while (abs(p2 - fat_pt)[1] / 140) > common_distance:
                p2[0] += 1
                p2[1] += 1
        elif (abs(p2 - fat_pt)[1] / 140 < min_distance):
            while (abs(p2 - fat_pt)[1] / 140) < common_distance:
                p2[0] += 1
                p2[1] -= 1

    elif angle_adj < 45 and current_fat_placement == "FAT_RIGHT":
        fat_pt = (fat_bbox[0] + fat_bbox[1]) / 2
        muscle_mid = (muscle_bbox[0] + muscle_bbox[3]) / 2

        if (abs(p2 - fat_pt)[0] / 140 > max_distance):
            print ("More than 6.5")
            while (abs(p2 - fat_pt)[1] / 140) > common_distance:
                p2[0] -= 1
                p2[1] -= 1
        elif (abs(p2 - fat_pt)[0] / 140 < min_distance):
            while (abs(p2 - fat_pt)[1] / 140) < common_distance:
                p2[0] -= 1
                p2[1] += 1

    elif angle_adj > 45 and current_fat_placement == "FAT_RIGHT":
        fat_pt = (fat_bbox[0] + fat_bbox[1]) / 2
        muscle_mid = (muscle_bbox[0] + muscle_bbox[3]) / 2

        if (abs(p2 - fat_pt)[0] / 140 > max_distance):
            while (abs(p2 - fat_pt)[0] / 140) > common_distance:
                p2[0] -= 1
                p2[1] += 1
        elif (abs(p2 - fat_pt)[0] / 140 < min_distance):
            while (abs(p2 - fat_pt)[0] / 140) < common_distance:
                p2[0] += 1
                p2[1] += 1

    elif angle_adj < 45 and current_fat_placement == "FAT_LEFT":
        fat_pt = (fat_bbox[2] + fat_bbox[3]) / 2
        muscle_mid = (muscle_bbox[1] + muscle_bbox[2]) / 2

        if (abs(p2 - fat_pt)[1] / 140 > max_distance):
            while (abs(p2 - fat_pt)[1] / 140) > common_distance:
                p2[0] += 1
                p2[1] += 1
        elif (abs(p2 - fat_pt)[1] / 140 < min_distance):
            while (abs(p2 - fat_pt)[1] / 140) < common_distance:
                p2[0] += 1
                p2[1] -= 1

    elif angle_adj > 45 and current_fat_placement == "FAT_LEFT":
        fat_pt = (fat_bbox[2] + fat_bbox[3]) / 2
        muscle_mid = (muscle_bbox[1] + muscle_bbox[2]) / 2

        if (abs(p2 - fat_pt)[0] / 140 > max_distance):
            while (abs(p2 - fat_pt)[0] / 140) > common_distance:
                p2[0] += 1
                p2[1] -= 1
        elif (abs(p2 - fat_pt)[0] / 140 < min_distance):
            while (abs(p2 - fat_pt)[0] / 140) < common_distance:
                p2[0] += 1
                p2[1] -= 1

    return p2, muscle_mid, fat_pt