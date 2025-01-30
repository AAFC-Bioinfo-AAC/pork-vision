# rotation.py

import numpy as np
import cv2
from utils.calculations import return_min_max_mask_coords, convert_back_to_xyxy
from utils.helpers import line_extender

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

