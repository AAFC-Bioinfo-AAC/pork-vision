# lines.py

import numpy as np
import cv2
from skimage import measure
from skimage.measure import points_in_poly
from skimage.draw import polygon2mask
from skimage.draw import line as skimage_line 
from ultralytics.data.utils import polygon2mask
from shapely.geometry import Polygon
from .helpers import rotation_detector_by_angle
from .calculations import return_measurements

def drawlines(contour_points, mask):
    """
    Draws lines on a binary mask based on the minimum and maximum contour points and calculates the area.

    Parameters:
    contour_points (numpy.ndarray): Array of contour points.
    mask (numpy.ndarray): Binary mask where the lines will be drawn.

    Returns:
    tuple: A tuple containing:
        - myarea (float): The calculated area of the polygon formed by the contour points.
        - h1_point (numpy.ndarray): The point with the minimum x-coordinate.
        - h2_point (numpy.ndarray): The point with the maximum x-coordinate.
        - v1_point (numpy.ndarray): The point with the minimum y-coordinate.
        - v2_point (numpy.ndarray): The point with the maximum y-coordinate.
        - with_line (numpy.ndarray): Binary mask with the lines drawn.
    """
    min_x = np.argmin(contour_points[:, 0])
    max_x = np.argmax(contour_points[:, 0])
    min_y = np.argmin(contour_points[:, 1])
    max_y = np.argmax(contour_points[:, 1])

    h1_point = contour_points[min_x, :]
    h2_point = contour_points[max_x, :]
    v1_point = contour_points[min_y, :]
    v2_point = contour_points[max_y, :]

    list_points2 = [(point[0], point[1]) for point in contour_points]
    list_points2.append((contour_points[0][0], contour_points[0][1]))

    mypolygon = Polygon(list_points2)
    myarea = mypolygon.area

    color = (0, 0, 255)
    thickness = 9

    new_mask = np.copy(mask)

    new_mask = cv2.line(
        new_mask,
        h1_point.astype("int32"),
        h2_point.astype("int32"),
        color,
        thickness,
    )

    return myarea, h1_point, h2_point, v1_point, v2_point, new_mask

def convert_contours_to_image(contours, orig_shape):
    """
    Converts YOLO inferred contours into a binary mask.

    Parameters:
    contours (list): List of contour points.
    orig_shape (tuple): Original shape of the image.

    Returns:
    numpy.ndarray: Binary mask with the contours drawn.
    """
    mask = polygon2mask(
        orig_shape,  # tuple
        [contours],  # input as list
        color=255,  # 8-bit binary
        downsample_ratio=1,
    )
    return mask

def find_nearest_contour_point(contour, p2):
    """
    Finds the contour point closest to a given point.

    Parameters:
    contour (numpy.ndarray): Array of contour points.
    p2 (numpy.ndarray): The point to find the nearest contour point to.

    Returns:
    numpy.ndarray: The nearest contour point.
    """
    distances = np.linalg.norm(contour - p2, axis=1)
    min_index = np.argmin(distances)
    return contour[min_index]

def check_mask_presence(current_image):
    """
    Checks whether there are masks inferred for each class (0 = muscle, 1 = fat).

    Parameters:
    current_image (object): The current image object containing YOLOv8 inference results.

    Returns:
    bool: True if both muscle and fat masks are present, False otherwise.
    """
    classes = current_image.boxes.cls.numpy().astype("int")
    confidences = current_image.boxes.conf.numpy()

    combo_list = np.column_stack((classes, confidences))

    muscle_present = False
    fat_present = False

    for i in combo_list[:, 0]:
        if i == 0:
            muscle_present = True
        elif i == 1:
            fat_present = True

    return muscle_present and fat_present

def mask_selector(current_image):
    """
    Selects the best masks for muscle and fat classes based on confidence scores.

    Parameters:
    current_image (object): The current image object containing YOLOv8 inference results.

    Returns:
    tuple: A tuple containing:
        - muscle_bbox (numpy.ndarray): Bounding box for the muscle mask.
        - muscle_mask (numpy.ndarray): Contour points for the muscle mask.
        - fat_bbox (numpy.ndarray): Bounding box for the fat mask.
        - fat_mask (numpy.ndarray): Contour points for the fat mask.
    """
    classes = current_image.boxes.cls.numpy().astype("int")
    confidences = current_image.boxes.conf.numpy()
    combo_list = np.column_stack((classes, confidences))

    confidence_muscle = 0
    confidence_fat = 0

    for j, cls in enumerate(combo_list[:, 0]):
        if cls == 0 and combo_list[j, 1] > confidence_muscle:
            confidence_muscle = combo_list[j, 1]
            muscle_bbox = current_image.boxes[j].xyxy
            muscle_mask = current_image.masks[j].xy
        elif cls == 1 and combo_list[j, 1] > confidence_fat:
            confidence_fat = combo_list[j, 1]
            fat_bbox = current_image.boxes[j].xyxy
            fat_mask = current_image.masks[j].xy

    return muscle_bbox[0], muscle_mask, fat_bbox[0], fat_mask

def line_to_fat(
    orientation,
    angle,
    min_h_muscle,
    max_h_muscle,
    min_v_muscle,
    max_v_muscle,
    rotated_fat_mask,
):
    """
    Finds which points of a line extended from the muscle fall onto the fat layer.

    Parameters:
    orientation (str): Orientation of the fat layer relative to the muscle layer.
    angle (float): Angle of rotation calculated by the ellipsoid method.
    min_h_muscle (numpy.ndarray): Minimum horizontal muscle point.
    max_h_muscle (numpy.ndarray): Maximum horizontal muscle point.
    min_v_muscle (numpy.ndarray): Minimum vertical muscle point.
    max_v_muscle (numpy.ndarray): Maximum vertical muscle point.
    rotated_fat_mask (numpy.ndarray): Binary mask of the rotated fat layer.

    Returns:
    tuple: A tuple containing:
        - p1 (numpy.ndarray): The starting point of the line segment.
        - p2 (numpy.ndarray): The ending point of the line segment.
        - max_fat_pt (numpy.ndarray): The point on the fat layer where the line intersects.
        - ld_depth (float): The calculated depth of the muscle.
        - ld_width (float): The calculated width of the muscle.
    """
    p1, p2, endpt_x, endpt_y, ld_depth, ld_width = rotation_detector_by_angle(
        orientation,
        angle,
        min_h_muscle,
        max_h_muscle,
        min_v_muscle,
        max_v_muscle,
    )
    discrete_line = list(zip(*skimage_line(*p2, *np.array([endpt_x, endpt_y]))))
    contours, _ = cv2.findContours(
        rotated_fat_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    pts = points_in_poly(discrete_line, contours[0].reshape(-1, 2))
    disc_line_fat = np.array(discrete_line)[pts]
    max_fat_pt = disc_line_fat[disc_line_fat.shape[0] - 1]

    return p1, p2, max_fat_pt, ld_depth, ld_width

def box_line(muscle_bbox, fat_bbox, img_rect, orientation):
    """
    Draws a line between the midpoints of the muscle and fat bounding boxes.

    Parameters:
    muscle_bbox (numpy.ndarray): Bounding box for the muscle.
    fat_bbox (numpy.ndarray): Bounding box for the fat.
    img_rect (numpy.ndarray): Image on which the line will be drawn.
    orientation (str): Orientation of the fat layer relative to the muscle layer.

    Returns:
    numpy.ndarray: Image with the line drawn.
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

    if orientation == "FAT_RIGHT":
        mid_pt_muscle = (muscle_bbox[2] + muscle_bbox[1]) / 2
        fat_connect = (fat_bbox[1][0], mid_pt_muscle[1])
    elif orientation == "FAT_LEFT":
        mid_pt_muscle = (muscle_bbox[3] + muscle_bbox[0]) / 2
        fat_connect = (fat_bbox[3][0], mid_pt_muscle[1])
    elif orientation == "FAT_BOTTOM":
        mid_pt_muscle = (muscle_bbox[2] + muscle_bbox[3]) / 2
        fat_connect = (mid_pt_muscle[0], fat_bbox[3][1])
    elif orientation == "FAT_TOP":
        mid_pt_muscle = (muscle_bbox[1] + muscle_bbox[0]) / 2
        fat_connect = (mid_pt_muscle[0], fat_bbox[1][1])

    fat_connect = np.array(fat_connect)
    img_rect = cv2.line(
        img_rect,
        mid_pt_muscle.astype("int32"),
        fat_connect.astype("int32"),
        (0, 255, 255),
        20,
    )
    return img_rect

def box_line_with_offset(muscle_bbox, fat_bbox, img_rect, orientation):
    """
    Draws a line between the midpoints of the muscle and fat bounding boxes with an offset.

    Parameters:
    muscle_bbox (numpy.ndarray): Bounding box for the muscle.
    fat_bbox (numpy.ndarray): Bounding box for the fat.
    img_rect (numpy.ndarray): Image on which the line will be drawn.
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

    if orientation == "FAT_RIGHT":
        mid_pt_muscle = (muscle_bbox[2] + muscle_bbox[1]) / 2
        fat_connect = (fat_bbox[1][0], mid_pt_muscle[1] - 400)
    elif orientation == "FAT_LEFT":
        mid_pt_muscle = (muscle_bbox[3] + muscle_bbox[0]) / 2
        fat_connect = (fat_bbox[3][0], mid_pt_muscle[1] + 400)
    elif orientation == "FAT_BOTTOM":
        mid_pt_muscle = (muscle_bbox[2] + muscle_bbox[3]) / 2
        fat_connect = (mid_pt_muscle[0] - 400, fat_bbox[3][1])
    elif orientation == "FAT_TOP":
        mid_pt_muscle = (muscle_bbox[1] + muscle_bbox[0]) / 2
        fat_connect = (mid_pt_muscle[0] + 400, fat_bbox[1][1])

    fat_connect = np.array(fat_connect)
    return mid_pt_muscle, fat_connect

def line_to_fat_box_method(mid_pt_muscle, fat_connect, fat_mask):
    """
    Connects a line from the muscle midpoint to the fat bounding box and finds the intersection point.

    Parameters:
    mid_pt_muscle (numpy.ndarray): Midpoint of the muscle bounding box.
    fat_connect (numpy.ndarray): Connection point on the fat bounding box.
    fat_mask (numpy.ndarray): Binary mask of the fat layer.

    Returns:
    numpy.ndarray: The point on the fat layer where the line intersects.
    """
    discrete_line = list(zip(*skimage_line(*mid_pt_muscle, *fat_connect)))
    contours, _ = cv2.findContours(
        fat_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    pts = points_in_poly(discrete_line, contours[0].reshape(-1, 2))
    disc_line_fat = np.array(discrete_line)[pts]
    max_fat_pt = disc_line_fat[-1]

    return max_fat_pt

