# helpers.py

import numpy as np
from utils.calculations import return_measurements

def line_extender(p1, p2):
    """
    Extends a line segment defined by two points.

    Parameters:
    p1 (numpy.ndarray): The starting point of the line segment.
    p2 (numpy.ndarray): The ending point of the line segment.

    Returns:
    tuple: A tuple containing:
        - p1 (numpy.ndarray): The starting point of the line segment.
        - p2 (numpy.ndarray): The ending point of the line segment.
        - endpt_x (int): The x-coordinate of the extended endpoint.
        - endpt_y (int): The y-coordinate of the extended endpoint.
    """
    theta = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    endpt_x = int(p1[0] - 5000 * np.cos(theta))
    endpt_y = int(p1[1] - 5000 * np.sin(theta))

    return p1, p2, endpt_x, endpt_y

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