# helpers.py

import numpy as np

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
