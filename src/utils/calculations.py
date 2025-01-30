# calculations.py

import numpy as np
import math

def return_measurements(depth_1, depth_2, width_1, width_2):
    """
    Calculates the muscle depth and width based on given points.

    Parameters:
    depth_1 (tuple): First point for depth measurement.
    depth_2 (tuple): Second point for depth measurement.
    width_1 (tuple): First point for width measurement.
    width_2 (tuple): Second point for width measurement.

    Returns:
    tuple: A tuple containing:
        - muscle_depth (float): The calculated depth of the muscle.
        - muscle_width (float): The calculated width of the muscle.
    """
    muscle_depth = abs(math.dist(depth_1, depth_2))
    muscle_width = abs(math.dist(width_1, width_2))

    return muscle_depth, muscle_width

def calculcate_midpoint_muscle_box(muscle_bbox, fat_bbox, orientation):
    """
    Calculates the midpoint of the muscle bounding box and the connection point on the fat bounding box.

    Parameters:
    muscle_bbox (numpy.ndarray): Bounding box for the muscle.
    fat_bbox (numpy.ndarray): Bounding box for the fat.
    orientation (str): Orientation of the fat layer relative to the muscle layer.

    Returns:
    tuple: A tuple containing:
        - mid_pt_muscle (numpy.ndarray): Midpoint of the muscle bounding box.
        - fat_connect (numpy.ndarray): Connection point on the fat bounding box.
    """
    muscle_bbox = np.array(
        (
            (muscle_bbox[0], muscle_bbox[1]),
            (muscle_bbox[2], muscle_bbox[1]),
            (muscle_bbox[2], muscle_bbox[3]),
            (muscle_bbox[0], muscle_bbox[3]),
        )
    )
    fat_bbox = np.array(
        (
            (fat_bbox[0], fat_bbox[1]),
            (fat_bbox[2], fat_bbox[1]),
            (fat_bbox[2], fat_bbox[3]),
            (fat_bbox[0], fat_bbox[3]),
        )
    )

    if orientation == "FAT_TOP":
        mid_pt_muscle = (muscle_bbox[1] + muscle_bbox[0]) / 2
        fat_connect = (mid_pt_muscle[0], fat_bbox[0][1])
    elif orientation == "FAT_BOTTOM":
        mid_pt_muscle = (muscle_bbox[3] + muscle_bbox[2]) / 2
        fat_connect = (mid_pt_muscle[0], fat_bbox[3][1])
    elif orientation == "FAT_RIGHT":
        mid_pt_muscle = (muscle_bbox[2] + muscle_bbox[1]) / 2
        fat_connect = (fat_bbox[1][0], mid_pt_muscle[1])
    elif orientation == "FAT_LEFT":
        mid_pt_muscle = (muscle_bbox[0] + muscle_bbox[3]) / 2
        fat_connect = (fat_bbox[0][0], mid_pt_muscle[1])

    return mid_pt_muscle, np.array(fat_connect)

def return_min_max_mask_coords(contours):
    """
    Returns the minimum and maximum coordinates in a contour list from any binary mask.

    Parameters:
    contours (numpy.ndarray): Array of contour points.

    Returns:
    list: A list of 4 points - horizontal minima and maxima, and vertical minima and maxima.
    """
    min_x = np.argmin(contours[:, 0])
    max_x = np.argmax(contours[:, 0])
    min_y = np.argmin(contours[:, 1])
    max_y = np.argmax(contours[:, 1])

    h1_point = contours[min_x, :]
    h2_point = contours[max_x, :]
    v1_point = contours[min_y, :]
    v2_point = contours[max_y, :]

    return [h1_point, h2_point, v1_point, v2_point]

def convert_back_to_xyxy(bbox):
    """
    Converts a bounding box to YOLO xyxy format.

    Parameters:
    bbox (numpy.ndarray): Bounding box coordinates.

    Returns:
    numpy.ndarray: Reformatted bounding box coordinates.
    """
    return np.array((bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]))

def bbox_reformatter(bbox):
    """
    Reformats a bounding box into a format required by either YOLOv8 or other software.

    Parameters:
    bbox (numpy.ndarray): Array of bounding box coordinates.

    Returns:
    numpy.ndarray: Reformatted bounding box coordinates.
    """
    min_x = bbox[0][0]
    min_y = bbox[0][1]
    max_x = bbox[2][0]
    max_y = bbox[2][1]

    return np.array([min_x, min_y, max_x, max_y])