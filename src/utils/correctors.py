# correctors.py

import cv2
import numpy as np
from skimage import draw, measure
from utils.calculations import calculcate_midpoint_muscle_box
from utils.rotation import rotate_box_line
from utils.helpers import line_extender
from utils.lines import find_nearest_contour_point

def correct_measurements(
    image_line,
    orientation,
    rotated_fat_box,
    rotated_muscle_box,
    rotated_fat_mask,
    p2,
    muscle_bbox,
    fat_bbox,
    angle,
    center,
):
    """
    Corrects the measurements of muscle and fat layers by drawing lines and adjusting points.

    Parameters:
    image_line (numpy.ndarray): The input image with lines to be drawn.
    orientation (str): Orientation of the fat layer relative to the muscle layer.
    rotated_fat_box (numpy.ndarray): Bounding box for the rotated fat layer.
    rotated_muscle_box (numpy.ndarray): Bounding box for the rotated muscle layer.
    rotated_fat_mask (numpy.ndarray): Binary mask of the rotated fat layer.
    p2 (numpy.ndarray): Point where the muscle connects to the fat.
    muscle_bbox (numpy.ndarray): Bounding box for the muscle.
    fat_bbox (numpy.ndarray): Bounding box for the fat.
    angle (float): Rotation angle in degrees.
    center (tuple): The center point (x, y) around which the image will be rotated.

    Returns:
    numpy.ndarray: Image with the corrected lines drawn.
    """
    mid_pt_muscle_boxes, fat_connect_boxes = calculcate_midpoint_muscle_box(
        muscle_bbox, fat_bbox, orientation
    )
    mid_pt_muscle, fat_connect = rotate_box_line(
        mid_pt_muscle_boxes, fat_connect_boxes, angle, center
    )

    new_p2, new_p1 = distance_corrector(
        image_line.copy(),
        orientation,
        rotated_fat_box,
        rotated_muscle_box,
        rotated_fat_mask.copy(),
        p2,
        angle,
        center,
    )
    new_p1, new_p2, end_x_method2, end_y_method2 = line_extender(
        new_p1, new_p2
    )
    new_p1, mid_pt_muscle, end_x_method1, end_y_method1 = line_extender(
        new_p1, mid_pt_muscle
    )

    max_fat_pt_method2 = line_to_fat_corrector(
        new_p1, end_x_method2, end_y_method2, rotated_fat_mask.copy()
    )
    max_fat_pt_method1 = line_to_fat_corrector(
        new_p1, end_x_method1, end_y_method1, rotated_fat_mask.copy()
    )

    image_line = cv2.line(
        image_line,
        new_p1.astype("int32"),
        mid_pt_muscle.astype("int32"),
        (0, 255, 255),
        10,
    )
    image_line = cv2.line(
        image_line,
        mid_pt_muscle.astype("int32"),
        max_fat_pt_method1,
        (255, 255, 0),
        10,
    )
    image_line = cv2.line(
        image_line,
        new_p1.astype("int32"),
        new_p2.astype("int32"),
        (255, 0, 255),
        10,
    )
    image_line = cv2.line(
        image_line,
        new_p2.astype("int32"),
        max_fat_pt_method2,
        (255, 255, 0),
        10,
    )

    return image_line

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

def rotation_detector_by_angle(
    current_fat_placement,
    angle,
    min_h_muscle,
    max_h_muscle,
    min_v_muscle,
    max_v_muscle,
):
    """
    Detects the orientation of the fat layer with respect to the muscle layer based on the angle of rotation.
    Extends a line from the muscle measurements to capture the depth of the fat layer.

    Parameters:
    current_fat_placement (str): The current placement of the fat layer ("FAT_TOP", "FAT_BOTTOM", "FAT_LEFT", "FAT_RIGHT").
    angle (float): The angle of rotation calculated by the ellipsoid method.
    min_h_muscle (numpy.ndarray): Minimum horizontal muscle point.
    max_h_muscle (numpy.ndarray): Maximum horizontal muscle point.
    min_v_muscle (numpy.ndarray): Minimum vertical muscle point.
    max_v_muscle (numpy.ndarray): Maximum vertical muscle point.

    Returns:
    tuple: A tuple containing:
        - p1 (numpy.ndarray): The starting point of the line segment.
        - p2 (numpy.ndarray): The ending point of the line segment.
        - endpt_x (int): The x-coordinate of the extended endpoint.
        - endpt_y (int): The y-coordinate of the extended endpoint.
        - ld_depth (float): The calculated depth of the muscle.
        - ld_width (float): The calculated width of the muscle.
    """
    if angle > 0:
        angle_adj = abs(90 - angle)
    else:
        angle_adj = 90 + angle

    if angle_adj < 45 and current_fat_placement == "FAT_BOTTOM":
        p1, p2, endpt_x, endpt_y = line_extender(min_v_muscle, max_v_muscle)
        ld_depth, ld_width = return_measurements(
            min_v_muscle, max_v_muscle, min_h_muscle, max_h_muscle
        )
    elif angle_adj > 45 and current_fat_placement == "FAT_BOTTOM":
        p1, p2, endpt_x, endpt_y = line_extender(min_h_muscle, max_h_muscle)
        ld_depth, ld_width = return_measurements(
            min_h_muscle, max_h_muscle, min_v_muscle, max_v_muscle
        )
    elif angle_adj < 45 and current_fat_placement == "FAT_TOP":
        p1, p2, endpt_x, endpt_y = line_extender(max_v_muscle, min_v_muscle)
        ld_depth, ld_width = return_measurements(
            max_v_muscle, min_v_muscle, min_h_muscle, max_h_muscle
        )
    elif angle_adj > 45 and current_fat_placement == "FAT_TOP":
        p1, p2, endpt_x, endpt_y = line_extender(max_h_muscle, min_h_muscle)
        ld_depth, ld_width = return_measurements(
            max_h_muscle, min_h_muscle, min_v_muscle, max_v_muscle
        )
    elif angle_adj < 45 and current_fat_placement == "FAT_RIGHT":
        p1, p2, endpt_x, endpt_y = line_extender(min_h_muscle, max_h_muscle)
        ld_depth, ld_width = return_measurements(
            min_h_muscle, max_h_muscle, min_v_muscle, max_v_muscle
        )
    elif angle_adj > 45 and current_fat_placement == "FAT_RIGHT":
        p1, p2, endpt_x, endpt_y = line_extender(max_v_muscle, min_v_muscle)
        ld_depth, ld_width = return_measurements(
            max_v_muscle, min_v_muscle, min_h_muscle, max_h_muscle
        )
    elif angle_adj < 45 and current_fat_placement == "FAT_LEFT":
        p1, p2, endpt_x, endpt_y = line_extender(max_h_muscle, min_h_muscle)
        ld_depth, ld_width = return_measurements(
            max_h_muscle, min_h_muscle, min_v_muscle, max_v_muscle
        )
    elif angle_adj > 45 and current_fat_placement == "FAT_LEFT":
        p1, p2, endpt_x, endpt_y = line_extender(min_v_muscle, max_v_muscle)
        ld_depth, ld_width = return_measurements(
            min_v_muscle, max_v_muscle, min_h_muscle, max_h_muscle
        )

    return p1, p2, endpt_x, endpt_y, ld_depth, ld_width

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