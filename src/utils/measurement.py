import numpy as np
import cv2
import math

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
        #print("Midline on the left")
        return "LEFT", leftmost
    else:
        #print("Midline on the right")
        return "RIGHT", rightmost

def measure_vertical_segment(muscle_mask, midline_side, cm_to_pixels=140):
    """
    Measures muscle depth by searching for the line segment within 5.5–6.5 cm from the midline
    that yields the greatest vertical (y-coordinate) muscle thickness, subject to the constraint
    that the line’s tilt is no more than 15° from vertical.
    
    The candidate line is defined as:
        x(y) = candidate_x + (y - mid_y) * tan(theta)
    where candidate_x is computed from the leftmost (or rightmost) x-coordinate of the muscle contour
    plus (or minus) an offset (in pixels) corresponding to the distance from the midline, and mid_y
    is the vertical center of the image. The tilt angle theta is allowed to vary between -15° and +15°.
    
    Parameters:
    - muscle_mask (numpy.ndarray): A binary image where nonzero pixels indicate muscle.
    - midline_side (str): Either "LEFT" or another value (e.g. "RIGHT") to indicate on which side
                          the midline is located. (For "LEFT", the candidate base is computed from the
                          leftmost contour x-coordinate; otherwise from the rightmost.)
    - cm_to_pixels (int, optional): Conversion factor from centimeters to pixels (default is 140).
    
    Returns:
    - tuple: ((x_bottom, y_bottom), (x_top, y_top)) representing the endpoints of the candidate line 
             with maximum vertical muscle depth. The bottom endpoint is the point with the largest y 
             value and the top endpoint with the smallest y value.
             If no valid muscle segment is found, returns (None, None).
    """
    # Select the largest contour (by area)
    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("measure_vertical_segment: No contours found.")
        return None, None
    muscle_contour = max(contours, key=cv2.contourArea)
    
    # Determine leftmost and rightmost x-coordinates from the contour
    leftmost_x = np.min(muscle_contour[:, :, 0])
    rightmost_x = np.max(muscle_contour[:, :, 0])
    
    # Defining the range of offsets (in pixels)
    min_offset = int(5.5 * cm_to_pixels)
    max_offset = int(6 * cm_to_pixels)
    
    height, width = muscle_mask.shape
    mid_y = height // 2  # Use the vertical center of the image as reference
    
    best_depth = 0
    best_endpoints = None

    # Iterate over candidate offsets (pixel distances) from the midline
    for offset in range(min_offset, max_offset + 1):
        # Compute the candidate base x-coordinate depending on the midline side.
        if midline_side.upper() == "LEFT":
            candidate_x = leftmost_x + offset
        else:
            candidate_x = rightmost_x - offset

        # Skip this candidate if the computed candidate_x is out of bounds.
        if candidate_x < 0 or candidate_x >= width:
            continue

        # Iterate over tilt angles from -5° to +5° (in 1° increments).
        for angle_deg in range(-5, 6):
            angle_rad = np.deg2rad(angle_deg)
            tan_angle = np.tan(angle_rad)
            
            candidate_line_points = []  # List to hold (x, y) points on the candidate line that are in the mask
            
            # Sample along the vertical direction (all possible y values)
            for y in range(height):
                # Define the x-coordinate along the candidate line.
                # When y == mid_y, x is exactly candidate_x.
                x = candidate_x + (y - mid_y) * tan_angle
                
                # Only consider the point if it lies within the image horizontally.
                if x < 0 or x >= width:
                    continue
                
                # Use nearest neighbor indexing for x.
                x_idx = int(round(x))
                
                # Check if the pixel on the candidate line is part of the muscle mask.
                if muscle_mask[y, x_idx] > 0:
                    candidate_line_points.append((x, y))
            
            # If no muscle pixels were found along this candidate line, move to the next candidate.
            if not candidate_line_points:
                continue

            # Determine the depth of the candidate line segment.
            y_coords = [pt[1] for pt in candidate_line_points]
            top_y = min(y_coords)
            bottom_y = max(y_coords)
            depth = bottom_y - top_y

            # Update the best candidate if this one has a greater vertical depth.
            if depth > best_depth:
                # Recompute the corresponding x coordinates for the endpoints using the candidate line equation.
                top_x = candidate_x + (top_y - mid_y) * tan_angle
                bottom_x = candidate_x + (bottom_y - mid_y) * tan_angle
                best_depth = depth
                best_endpoints = ((bottom_x, bottom_y), (top_x, top_y))

    if best_endpoints is None:
        print("measure_vertical_segment: No valid muscle segment found within specified range and tilt constraints.")
        return None, None

    return best_endpoints

def extend_vertical_line_to_fat(fat_mask, muscle_depth_line, step=1.0, max_iter=10000):
    """
    Measures the fat depth as a continuation of the muscle depth line.
    
    Starting immediately after the top endpoint of the muscle depth line
    (i.e. the point where the muscle measurement ended), this function
    extends the line along the same trajectory (same angle) until the fat mask
    is encountered and then until the fat mask ends. The fat depth is measured
    from the first encountered fat pixel along the line to the last consecutive
    fat pixel.
    
    Parameters:
    - fat_mask (numpy.ndarray): Binary mask of the fat (nonzero where fat exists).
    - muscle_depth_line (tuple): A tuple of two endpoints ((x_bottom, y_bottom), (x_top, y_top))
      defining the muscle depth line. The top endpoint is assumed to be where the muscle stops.
    - step (float, optional): The step size in pixels to sample along the line (default is 1.0).
    - max_iter (int, optional): Maximum number of steps to avoid infinite loops (default is 10000).
    
    Returns:
    - tuple: ((x_fat_start, y_fat_start), (x_fat_end, y_fat_end)) representing the fat depth line,
      where (x_fat_start, y_fat_start) is the first fat pixel encountered along the line and
      (x_fat_end, y_fat_end) is the last pixel that remains in the fat mask.
      If no fat pixels are encountered along the line, returns (None, None).
    """
    muscle_bottom, muscle_top = muscle_depth_line

    # Compute the unit direction vector from muscle_bottom to muscle_top.
    dx = muscle_top[0] - muscle_bottom[0]
    dy = muscle_top[1] - muscle_bottom[1]
    norm = math.sqrt(dx * dx + dy * dy)
    if norm == 0:
        print("extend_line_to_fat: Muscle depth line has zero length.")
        return None, None
    unit_dx = dx / norm
    unit_dy = dy / norm

    height, width = fat_mask.shape

    # Starting from the muscle top, move one step along the line.
    current_point = (muscle_top[0] + unit_dx * step, muscle_top[1] + unit_dy * step)

    iter_count = 0
    entry_point = None  # Will store the first point that is in the fat mask

    # Step 1: Advance along the line until we find the first fat pixel.
    while iter_count < max_iter:
        new_x = int(round(current_point[0]))
        new_y = int(round(current_point[1]))
        # Break if out of image bounds.
        if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
            break
        if fat_mask[new_y, new_x] > 0:
            entry_point = (current_point[0], current_point[1])
            break
        # Continue stepping.
        current_point = (current_point[0] + unit_dx * step, current_point[1] + unit_dy * step)
        iter_count += 1

    if entry_point is None:
        print("extend_line_to_fat: No fat region encountered along the line.")
        return None, None

    # Step 2: Continue along the line as long as the fat mask is detected.
    fat_start = (int(round(entry_point[0])), int(round(entry_point[1])))
    last_fat_point = entry_point
    # Reset the iteration counter if needed.
    iter_count = 0
    while iter_count < max_iter:
        next_point = (last_fat_point[0] + unit_dx * step, last_fat_point[1] + unit_dy * step)
        next_x = int(round(next_point[0]))
        next_y = int(round(next_point[1]))
        if next_x < 0 or next_x >= width or next_y < 0 or next_y >= height:
            break
        if fat_mask[next_y, next_x] > 0:
            last_fat_point = next_point
            iter_count += 1
        else:
            break

    fat_end = (int(round(last_fat_point[0])), int(round(last_fat_point[1])))
    return fat_start, fat_end