# visualizations.py 

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

