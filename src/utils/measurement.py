import numpy as np
import cv2

def measure_longest_horizontal_segment(muscle_mask):
    """
    Finds the longest horizontal segment in the muscle mask and returns its start & end points.

    Returns:
    - tuple: ((x1, y1), (x2, y2)) representing the longest horizontal segment.
    """
    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None  # Return None if no contour is found

    muscle_contour = max(contours, key=cv2.contourArea)
    leftmost = tuple(muscle_contour[muscle_contour[:, :, 0].argmin()][0])
    rightmost = tuple(muscle_contour[muscle_contour[:, :, 0].argmax()][0])

    return leftmost, rightmost  # Return exact coordinates

def find_midline_using_fat_extremes(fat_mask):
    """
    Determines the midline (bone side) using the vertical position of the fat mask's leftmost and rightmost points.

    The higher point (lower Y value) is the bone side.

    Parameters:
    - fat_mask (numpy.ndarray): Binary mask of the fat.

    Returns:
    - str: "LEFT" if the midline is on the left side, "RIGHT" if on the right.
    - tuple: Coordinates of the detected midline point.
    """
    contours, _ = cv2.findContours(fat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found in fat mask.")
        return None, None

    fat_contour = max(contours, key=cv2.contourArea)
    fat_contour = np.squeeze(fat_contour)  # Fix potential shape issue

    # Find leftmost and rightmost fat points
    leftmost = tuple(fat_contour[np.argmin(fat_contour[:, 0])])  # (x, y)
    rightmost = tuple(fat_contour[np.argmax(fat_contour[:, 0])])  # (x, y)

    # The higher (smaller Y) point is the midline (bone side)
    if leftmost[1] < rightmost[1]:  # Leftmost is higher up
        print("Midline on the left")
        return "LEFT", leftmost
    else:
        print("Midline on the right")
        return "RIGHT", rightmost

def measure_vertical_segment(muscle_mask, midline_side, cm_to_pixels=140):
    """
    Measures muscle depth using a fixed x-coordinate at 7 cm from the midline (bone side).

    Returns:
    - tuple: ((x, y1), (x, y2)) representing the muscle depth line.
    """
    depth_offset = int(7 * cm_to_pixels)  # Convert cm to pixels

    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("measure_vertical_segment: No contours found.")
        return None, None

    muscle_contour = max(contours, key=cv2.contourArea)
    
    # Compute leftmost and rightmost x-coordinates
    leftmost_x = np.min(muscle_contour[:, :, 0])
    rightmost_x = np.max(muscle_contour[:, :, 0])

    # Offset direction based on midline side
    start_x = leftmost_x + depth_offset if midline_side == "LEFT" else rightmost_x - depth_offset

    # Ensure x is within valid image bounds
    if start_x < 0 or start_x >= muscle_mask.shape[1]:
        print(f"measure_vertical_segment: Computed x={start_x} is out of bounds. Skipping measurement.")
        return None, None

    # Extract the muscle mask column at start_x
    column_pixels = muscle_mask[:, start_x]

    # Find nonzero (muscle) pixel locations
    muscle_pixels = np.where(column_pixels > 0)[0]  # Get y-values where muscle exists

    if muscle_pixels.size == 0:
        print(f"measure_vertical_segment: No muscle detected at x = {start_x}. Skipping measurement.")
        return None, None

    # Get the min (top) and max (bottom) y-coordinates of the muscle in this column
    top_muscle_y = np.min(muscle_pixels)
    bottom_muscle_y = np.max(muscle_pixels)

    return (start_x, bottom_muscle_y), (start_x, top_muscle_y)

def extend_vertical_line_to_fat(fat_mask, muscle_depth_x):
    """
    Measures the fat depth at the same x-coordinate used for muscle depth.

    Parameters:
    - fat_mask (numpy.ndarray): Binary mask of the fat (1 where fat exists, 0 elsewhere).
    - muscle_depth_x (int): The x-coordinate used for muscle depth measurement.

    Returns:
    - tuple: ((x, y1), (x, y2)) representing the fat depth line.
    """
    if muscle_depth_x is None or muscle_depth_x < 0 or muscle_depth_x >= fat_mask.shape[1]:
        print("extend_vertical_line_to_fat: Invalid x-coordinate. Skipping measurement.")
        return None, None

    # Extract the fat mask column at muscle_depth_x
    column_pixels = fat_mask[:, muscle_depth_x]

    # Find nonzero (fat) pixel locations
    fat_pixels = np.where(column_pixels > 0)[0]  # Get y-values where fat exists

    if fat_pixels.size == 0:
        print(f"extend_vertical_line_to_fat: No fat detected at x = {muscle_depth_x}. Skipping measurement.")
        return None, None

    # Get the min (top) and max (bottom) y-coordinates of the fat in this column
    top_fat_y = np.min(fat_pixels)
    bottom_fat_y = np.max(fat_pixels)

    return (muscle_depth_x, bottom_fat_y), (muscle_depth_x, top_fat_y)