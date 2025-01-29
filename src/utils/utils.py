from ultralytics import YOLO
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import measure, draw
import math
from shapely.geometry import Polygon
import ellipse
from ultralytics.data.utils import polygon2mask
import pandas as pd
from csv import reader
from tabulate import tabulate


# Author: Fatima Davelouis


def fit_ellipse_v1(cnt_points_arr):
    """
    Fits an ellipse to the given contour points using least squares method.

    Parameters:
    cnt_points_arr (numpy.ndarray): Array of contour points.

    Returns:
    tuple: A tuple containing:
        - reg (ellipse.LsqEllipse): The fitted ellipse object.
        - center (tuple): The center coordinates of the ellipse.
        - width (float): The semimajor axis (horizontal dimension) of the ellipse.
        - height (float): The semiminor axis (vertical dimension) of the ellipse.
        - angle (float): The tilt angle of the ellipse in degrees.
    """
    reg = ellipse.LsqEllipse().fit(cnt_points_arr)
    center, width, height, phi = reg.as_parameters()
    angle = np.rad2deg(phi)

    return reg, center, width, height, angle


def fit_ellipse_v2(cnt_points):
    """
    Fits an ellipse to the given contour points using least squares method.

    Parameters:
    cnt_points (list or numpy.ndarray): List or array of contour points.

    Returns:
    tuple: A tuple containing:
        - alphas (numpy.ndarray): The coefficients of the fitted ellipse.
        - cnt_points_arr (numpy.ndarray): Reshaped array of contour points.
        - X (numpy.ndarray): X coordinates of contour points.
        - Y (numpy.ndarray): Y coordinates of contour points.
        - Z (numpy.ndarray): Calculated Z values for the ellipse equation.
    """
    cnt_points_arr = np.array(cnt_points).reshape(-1, 2, 1)
    X = cnt_points_arr[:, 0]
    Y = cnt_points_arr[:, 1]
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    b = np.ones_like(X)
    alphas = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()
    Z = (
        alphas[0] * X**2
        + alphas[1] * X * Y
        + alphas[2] * Y**2
        + alphas[3] * X
        + alphas[4] * Y
    )

    return alphas, cnt_points_arr, X, Y, Z


def plot_ellipse(cnt_points_arr, alphas=np.array([]), img_show=False):
    """
    Plots an ellipse based on the given contour points and coefficients.

    Parameters:
    cnt_points_arr (numpy.ndarray): Array of contour points.
    alphas (numpy.ndarray): Coefficients of the fitted ellipse equation. Default is an empty array.
    img_show (bool): Flag to display the plot. Default is False.

    Returns:
    numpy.ndarray: Array of ellipse coordinates.
    """
    ellipse_coord = None

    X = cnt_points_arr[:, 0]
    Y = cnt_points_arr[:, 1]

    X_min = np.min(X)
    X_max = np.max(X)
    Y_min = np.min(Y)
    Y_max = np.max(Y)

    x_coord = np.linspace(X_min - 10, X_max + 10, 300)
    y_coord = np.linspace(Y_min - 10, Y_max + 10, 300)

    X_coord, Y_coord = np.meshgrid(x_coord, y_coord)

    if len(alphas) > 0:
        Z_coord = (
            alphas[0] * X_coord**2
            + alphas[1] * X_coord * Y_coord
            + alphas[2] * Y_coord**2
            + alphas[3] * X_coord
            + alphas[4] * Y_coord
        )
        tt = plt.contour(
            X_coord, Y_coord, Z_coord, levels=[1], colors="r", linewidths=2
        )
        ellipse_coord = np.array(
            [
                [int(round(item[0])), int(round(item[1]))]
                for item in tt.allsegs[0][0]
            ]
        )

    if img_show:
        plt.scatter(X, Y, label="Data Points")
        if len(alphas) > 0:
            plt.contour(
                X_coord, Y_coord, Z_coord, levels=[1], colors="r", linewidths=2
            )
        plt.show()

    return ellipse_coord


def plot_polygon(mask_binary, contour_points, color=(255, 0, 0), thickness=1):
    """
    Draws a polygon on a binary mask based on the given contour points.

    Parameters:
    mask_binary (numpy.ndarray): Binary mask where the polygon will be drawn.
    contour_points (numpy.ndarray): Array of contour points.
    color (tuple): Color of the polygon. Default is (255, 0, 0).
    thickness (int): Thickness of the polygon lines. Default is 1.

    Returns:
    numpy.ndarray: Binary mask with the polygon drawn.
    """
    polycontour_mask = np.zeros(
        (mask_binary.shape[0], mask_binary.shape[1]), dtype="uint8"
    )
    pts = np.array(contour_points, np.int32).reshape((-1, 1, 2))
    isClosed = True
    cv2.polylines(polycontour_mask, [pts], isClosed, color, thickness)
    return polycontour_mask


def create_fitting_ellipse(img, mask, cnt_points):
    """
    Fits an ellipse onto the muscle mask.

    Parameters:
    img (numpy.ndarray): The input image.
    mask (numpy.ndarray): The binary muscle mask.
    cnt_points (list or numpy.ndarray): Contour points of the muscle mask.

    Returns:
    tuple: A tuple containing:
        - center (tuple): The center coordinates of the ellipse.
        - angle (float): The angle of ellipse inclination with respect to the x-axis.
        - img_color_ellipse_overlay (numpy.ndarray): Image with the ellipse overlayed.
    """
    cnt_points_arr = cnt_points[0]
    alphas, cnt_points_arr, X, Y, Z = fit_ellipse_v2(cnt_points)
    ellipse_coord = plot_ellipse(cnt_points_arr, alphas=alphas, img_show=False)
    reg, center, width, height, angle = fit_ellipse_v1(ellipse_coord)
    polycontour_ellipse_mask = plot_polygon(
        mask, ellipse_coord, color=(255, 0, 0), thickness=5
    )
    polycontour_ellipse_mask_3c = np.stack(
        [polycontour_ellipse_mask] * 3, axis=-1
    )
    img_color_ellipse_overlay = cv2.addWeighted(
        img, 0.1, polycontour_ellipse_mask_3c, 1, 0
    )

    return center, angle, img_color_ellipse_overlay


# Author: Edward Yakubovitch


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


def rotation_detector(image_aspectratio, muscle_contour, fat_contour):
    """
    Detects the rotation of an image based on the relationship between muscle and fat contours.

    Parameters:
    image_aspectratio (tuple): Aspect ratio of the image (width, height).
    muscle_contour (numpy.ndarray): Contour points of the muscle.
    fat_contour (numpy.ndarray): Contour points of the fat.

    Returns:
    str: Orientation of the fat layer relative to the muscle layer. Possible values are:
         "FAT_TOP", "FAT_BOTTOM", "FAT_LEFT", "FAT_RIGHT".
    """
    img_ar = image_aspectratio
    list_muscle = return_min_max_mask_coords(muscle_contour)
    list_fat = return_min_max_mask_coords(fat_contour)

    if img_ar[0] < img_ar[1]:  # Vertical orientation
        if list_muscle[2][1] > list_fat[2][1]:
            return "FAT_TOP"
        else:
            return "FAT_BOTTOM"
    else:  # Horizontal orientation
        if list_muscle[0][0] > list_fat[0][0]:
            return "FAT_LEFT"
        else:
            return "FAT_RIGHT"


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

    with_line = cv2.line(
        mask,
        h1_point.astype("int32"),
        h2_point.astype("int32"),
        color,
        thickness,
    )

    return myarea, h1_point, h2_point, v1_point, v2_point, with_line


def rotate_image(img, bbox, angle, center):
    """
    Rotates the image according to a rotation matrix influenced by the angle calculated during the ellipsoid fitting.

    Parameters:
    img (numpy.ndarray): The input image.
    bbox (numpy.ndarray): Bounding box coordinates in the format [x1, y1, x2, y2].
    angle (float): Rotation angle in degrees, positive values for counter-clockwise rotation.
    center (tuple): The center point (x, y) around which the image will be rotated.

    Returns:
    tuple: A tuple containing:
        - img_rotated (numpy.ndarray): The rotated image.
        - bb_rotated (numpy.ndarray): The rotated bounding box coordinates.
        - bbox (numpy.ndarray): The original bounding box coordinates.
    """
    bbox = np.array(
        [
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
        ]
    )

    if angle > 0:
        rotMat = cv2.getRotationMatrix2D(center, abs(90 - angle), 1.0)
    else:
        rotMat = cv2.getRotationMatrix2D(center, 90 + angle, 1.0)

    img_rotated = cv2.warpAffine(img, rotMat, img.shape[1::-1])
    bb_rotated = np.dot(rotMat, np.vstack((bbox.T, np.ones(4)))).T

    return img_rotated, bb_rotated, bbox


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
    discrete_line = list(zip(*draw.line(*p2, *np.array([endpt_x, endpt_y]))))
    contours, _ = cv2.findContours(
        rotated_fat_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    pts = measure.points_in_poly(discrete_line, contours[0].reshape(-1, 2))
    disc_line_fat = np.array(discrete_line)[pts]
    max_fat_pt = disc_line_fat[disc_line_fat.shape[0] - 1]

    return p1, p2, max_fat_pt, ld_depth, ld_width


def reverse_orientation(orientation):
    """
    Reverses the orientation of the fat layer.

    Parameters:
    orientation (str): Current orientation of the fat layer.

    Returns:
    str: Reversed orientation of the fat layer.
    """
    if orientation == "FAT_RIGHT":
        return "FAT_LEFT"
    elif orientation == "FAT_LEFT":
        return "FAT_RIGHT"
    elif orientation == "FAT_BOTTOM":
        return "FAT_TOP"
    elif orientation == "FAT_TOP":
        return "FAT_BOTTOM"


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
    discrete_line = list(zip(*draw.line(*mid_pt_muscle, *fat_connect)))
    contours, _ = cv2.findContours(
        fat_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    pts = measure.points_in_poly(discrete_line, contours[0].reshape(-1, 2))
    disc_line_fat = np.array(discrete_line)[pts]
    max_fat_pt = disc_line_fat[-1]

    return max_fat_pt


def convert_back_to_xyxy(bbox):
    """
    Converts a bounding box to YOLO xyxy format.

    Parameters:
    bbox (numpy.ndarray): Bounding box coordinates.

    Returns:
    numpy.ndarray: Reformatted bounding box coordinates.
    """
    return np.array((bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]))


def rotate_box_line(mid_pt_muscle, fat_connect, angle, center):
    """
    Rotates a line (created as a pseudo-box) according to an angle derived from the ellipsoid method.

    Parameters:
    mid_pt_muscle (numpy.ndarray): Midpoint of the muscle bounding box.
    fat_connect (numpy.ndarray): Connection point on the fat bounding box.
    angle (float): Rotation angle in degrees.
    center (tuple): The center point (x, y) around which the image will be rotated.

    Returns:
    tuple: A tuple containing:
        - mid_pt_muscle (numpy.ndarray): Rotated midpoint of the muscle bounding box.
        - fat_connect (numpy.ndarray): Rotated connection point on the fat bounding box.
    """
    bbox = np.concatenate([mid_pt_muscle, fat_connect])
    bbox = np.array(
        (
            (bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]),
        )
    )
    rotMat = cv2.getRotationMatrix2D(center, abs(90 - angle), 1.0)
    bb_rotated = np.dot(rotMat, np.vstack((bbox.T, np.ones(4)))).T
    mid_pt_muscle = convert_back_to_xyxy(bb_rotated)[:2]
    fat_connect = convert_back_to_xyxy(bb_rotated)[2:4]

    return mid_pt_muscle, fat_connect


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


def draw_rotated_boxes_lines(
    img, muscle_bbox, fat_bbox, mid_pt_muscle, fat_connect
):
    """
    Draws rotated bounding boxes and lines connecting the midpoints.

    Parameters:
    img (numpy.ndarray): The input image.
    muscle_bbox (numpy.ndarray): Bounding box for the muscle.
    fat_bbox (numpy.ndarray): Bounding box for the fat.
    mid_pt_muscle (numpy.ndarray): Midpoint of the muscle bounding box.
    fat_connect (numpy.ndarray): Connection point on the fat bounding box.

    Returns:
    None
    """
    pt1_muscle = muscle_bbox[0:2].astype("int32")
    pt2_muscle = muscle_bbox[2:4].astype("int32")
    pt1_fat = fat_bbox[0:2].astype("int32")
    pt2_fat = fat_bbox[2:4].astype("int32")

    img_rect = cv2.rectangle(
        img.copy(), pt1_muscle, pt2_muscle, (255, 0, 0), 20
    )
    img_rect = cv2.rectangle(img_rect, pt1_fat, pt2_fat, (0, 0, 255), 20)
    img_rect = cv2.line(
        img_rect,
        mid_pt_muscle.astype("int32"),
        fat_connect.astype("int32"),
        (0, 255, 255),
        20,
    )

    plt.imshow(img_rect)
    plt.show()


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
        (0, 255, 255), # Yellow 
        10,
    )
    image_line = cv2.line(
        image_line,
        mid_pt_muscle.astype("int32"),
        max_fat_pt_method1,
        (255, 255, 0), # Teal
        10,
    )
    image_line = cv2.line(
        image_line,
        new_p1.astype("int32"),
        new_p2.astype("int32"),
        (255, 0, 255), # Violet
        10,
    )
    image_line = cv2.line(
        image_line,
        new_p2.astype("int32"),
        max_fat_pt_method2,
        (255, 255, 0), # Teal
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
