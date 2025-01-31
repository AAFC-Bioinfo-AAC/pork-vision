# rotation.py

import numpy as np
import cv2


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

