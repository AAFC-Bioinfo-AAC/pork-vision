from utils.imports import *
import math

def measure_longest_horizontal_segment(muscle_mask, rotation_angle, step=1):
    """
    Finds the longest segment across the muscle mask along the width direction,
    defined as the direction perpendicular to the muscles rotation angle.
    
    Parameters:
    - muscle_mask (numpy.ndarray): A binary image where nonzero pixels indicate muscle.
    - rotation_angle (float): The rotation angle in degrees, as computed from get_muscle_rotation_angle.
    - step (int, optional): Step size (in pixels along the major axis) when iterating candidate base points.
    
    Returns:
    - tuple: ((x_start, y_start), (x_end, y_end)) representing the endpoints of the measured segment.
             The endpoints are ordered such that the first has the lower projection value along the width direction.
             Returns (None, None) if no valid segment is found.
    """
    # Find contours and select the largest by area.
    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("measure_longest_horizontal_segment: No contours found.")
        return None, None

    muscle_contour = max(contours, key=cv2.contourArea)

    # Compute the center of the muscle contour.
    contour_points = muscle_contour.reshape(-1, 2)
    center = np.mean(contour_points, axis=0)

    # Convert the rotation angle to radians and compute unit vector along the muscle's long axis.
    angle_rad = np.deg2rad(rotation_angle)
    d = (np.cos(angle_rad), np.sin(angle_rad))
    
    # Compute the perpendicular direction (width direction).
    d_perp = (-np.sin(angle_rad), np.cos(angle_rad))
    # Ensure the x-component of d_perp is nonnegative.
    if d_perp[0] < 0:
        d_perp = (-d_perp[0], -d_perp[1])

    # Project contour points onto the major axis (d) to determine candidate base positions.
    # The projection scalar for a point p is: (p - center) dot d.
    projections = [(pt[0] - center[0]) * d[0] + (pt[1] - center[1]) * d[1] for pt in contour_points]
    min_t = int(np.floor(min(projections)))
    max_t = int(np.ceil(max(projections)))

    height, width = muscle_mask.shape

    best_distance = 0
    best_endpoints = None

    # Iterate over candidate base positions along the muscle's long axis.
    for t in range(min_t, max_t + 1, step):
        # Candidate base is moved along the major axis from the center.
        candidate_base = (center[0] + t * d[0], center[1] + t * d[1])

        # Walk in the positive direction along d_perp.
        t_pos = 0
        while True:
            t_candidate = t_pos + 1
            x_new = candidate_base[0] + t_candidate * d_perp[0]
            y_new = candidate_base[1] + t_candidate * d_perp[1]
            x_idx = int(round(x_new))
            y_idx = int(round(y_new))
            if x_idx < 0 or x_idx >= width or y_idx < 0 or y_idx >= height:
                break
            if muscle_mask[y_idx, x_idx] == 0:
                break
            t_pos = t_candidate

        # Walk in the negative direction along d_perp.
        t_neg = 0
        while True:
            t_candidate = t_neg - 1
            x_new = candidate_base[0] + t_candidate * d_perp[0]
            y_new = candidate_base[1] + t_candidate * d_perp[1]
            x_idx = int(round(x_new))
            y_idx = int(round(y_new))
            if x_idx < 0 or x_idx >= width or y_idx < 0 or y_idx >= height:
                break
            if muscle_mask[y_idx, x_idx] == 0:
                break
            t_neg = t_candidate

        # Total segment length is the difference between the positive and negative extents.
        distance = t_pos - t_neg
        if distance > best_distance:
            endpoint1 = (candidate_base[0] + t_neg * d_perp[0],
                         candidate_base[1] + t_neg * d_perp[1])
            endpoint2 = (candidate_base[0] + t_pos * d_perp[0],
                         candidate_base[1] + t_pos * d_perp[1])
            best_distance = distance
            best_endpoints = (endpoint1, endpoint2)

    if best_endpoints is None:
        print("measure_longest_horizontal_segment: No valid segment found.")
        return None, None

    # Order the endpoints by their projection onto d_perp so that the first is the 'start'.
    proj1 = best_endpoints[0][0] * d_perp[0] + best_endpoints[0][1] * d_perp[1]
    proj2 = best_endpoints[1][0] * d_perp[0] + best_endpoints[1][1] * d_perp[1]
    if proj1 <= proj2:
        start_point, end_point = best_endpoints
    else:
        start_point, end_point = best_endpoints[1], best_endpoints[0]

    return start_point, end_point

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

def get_muscle_rotation_angle(muscle_mask):
    """
    Computes the rotation angle of the muscle based on the largest contour in the muscle mask.
    
    This function finds all external contours in the binary muscle mask, selects the largest by area,
    and fits an ellipse to it using cv2.fitEllipse. The angle of the fitted ellipse (its major axis relative
    to the horizontal) is returned as the muscle’s rotation angle.
    
    Parameters:
    - muscle_mask (numpy.ndarray): A binary image where nonzero pixels indicate muscle.
    
    Returns:
    - float or None: The rotation angle in degrees. The angle is defined as the angle between the ellipse's
      major axis and the horizontal axis. (For example, 0° means the ellipse is horizontally oriented,
      and 90° means it is vertically oriented.) If no valid contour or ellipse is found, returns None.
    """
    # Find all external contours in the mask.
    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("get_muscle_rotation_angle: No contours found in the muscle mask.")
        return None

    # Select the largest contour by area.
    largest_contour = max(contours, key=cv2.contourArea)

    # Ensure there are enough points to fit an ellipse (at least 5 are required).
    if len(largest_contour) < 5:
        print("get_muscle_rotation_angle: Not enough contour points to fit an ellipse.")
        return None

    # Fit an ellipse to the largest contour.
    ellipse_fit = cv2.fitEllipse(largest_contour)
    # cv2.fitEllipse returns a tuple:
    #   ((center_x, center_y), (major_axis_length, minor_axis_length), angle)
    # The angle is measured in degrees between the ellipse's major axis and the horizontal.
    angle = ellipse_fit[2]
    
    return angle

def measure_vertical_segment(muscle_mask, midline_side, rotation_angle, cm_to_pixels=140):
    """
    Finds the absolute longest segment through the muscle mask along the known rotation angle 
    of the muscle. The search is restricted to a candidate base region defined by an offset 
    range (in pixels, converted from cm) from the midline.
    
    The candidate base is computed as:
        - For midline_side "LEFT": candidate_base_x = leftmost contour x + offset
        - For midline_side "RIGHT": candidate_base_x = rightmost contour x - offset
    The candidate base point is set to (candidate_base_x, mid_y), where mid_y is the vertical
    center of the image.
    
    Then, using the given rotation_angle (in degrees), the function “walks” along the line
    defined by:
        L(t) = candidate_base + t * d,  where d = (cos(angle), sin(angle))
    in both directions until the mask is exited. The candidate with the maximum distance
    between its two endpoints is returned.
    
    Parameters:
    - muscle_mask (numpy.ndarray): A binary image where nonzero pixels indicate muscle.
    - midline_side (str): "LEFT" or "RIGHT" to indicate on which side the carcass midline lies.
                          (For "LEFT", the candidate base is computed from the leftmost contour x;
                           for "RIGHT", from the rightmost.)
    - rotation_angle (float): The rotation angle (in degrees) of the muscle (as determined by
                              your ellipse-fitting code).
    - cm_to_pixels (int, optional): Conversion factor from centimeters to pixels (default is 140).
    
    Returns:
    - tuple: ((x_bottom, y_bottom), (x_top, y_top)) representing the endpoints of the segment
             with maximum distance. The endpoint with the larger y value is designated as the bottom.
             If no valid segment is found, returns (None, None).
    """
    # Find contours and select the largest by area.
    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("measure_vertical_segment: No contours found.")
        return None, None
    muscle_contour = max(contours, key=cv2.contourArea)

    # Determine the extreme x-values of the contour.
    leftmost_x = np.min(muscle_contour[:, :, 0])
    rightmost_x = np.max(muscle_contour[:, :, 0])

    # Convert cm offsets to pixels.
    min_offset = int(4 * cm_to_pixels)
    max_offset = int(5.75 * cm_to_pixels)

    height, width = muscle_mask.shape

    # Determine candidate base x-range based on the midline side.
    if midline_side.upper() == "LEFT":
        base_min = leftmost_x + min_offset
        base_max = leftmost_x + max_offset
    else:  # "RIGHT"
        # For the right side we want an increasing range:
        base_min = rightmost_x - max_offset
        base_max = rightmost_x - min_offset

    # Clamp candidate x values to be within the image bounds.
    base_min = max(0, min(base_min, width - 1))
    base_max = max(0, min(base_max, width - 1))

    # Use the vertical center of the image as the base y-coordinate.
    mid_y = height // 2

    # Convert rotation angle to radians and compute the unit direction vector.
    angle_rad = np.deg2rad(rotation_angle)
    d = (np.cos(angle_rad), np.sin(angle_rad))

    best_distance = 0
    best_endpoints = None


    # Iterate over candidate base x positions in the valid range.
    for candidate_x in range(base_min, base_max + 1):
        candidate_base = (candidate_x, mid_y)

        # Walk in the positive t direction.
        t_pos = 0
        while True:
            t = t_pos + 1
            x_new = candidate_base[0] + t * d[0]
            y_new = candidate_base[1] + t * d[1]
            x_idx = int(round(x_new))
            y_idx = int(round(y_new))
            if x_idx < 0 or x_idx >= width or y_idx < 0 or y_idx >= height:
                break
            if muscle_mask[y_idx, x_idx] == 0:
                break
            t_pos = t

        # Walk in the negative t direction.
        t_neg = 0
        while True:
            t = t_neg - 1
            x_new = candidate_base[0] + t * d[0]
            y_new = candidate_base[1] + t * d[1]
            x_idx = int(round(x_new))
            y_idx = int(round(y_new))
            if x_idx < 0 or x_idx >= width or y_idx < 0 or y_idx >= height:
                break
            if muscle_mask[y_idx, x_idx] == 0:
                break
            t_neg = t

        # Total length along this candidate line is the difference between the two extremes.
        distance = t_pos - t_neg  # (t_neg is negative so gives sum)
        if distance > best_distance:
            endpoint1 = (candidate_base[0] + t_neg * d[0], candidate_base[1] + t_neg * d[1])
            endpoint2 = (candidate_base[0] + t_pos * d[0], candidate_base[1] + t_pos * d[1])
            best_distance = distance
            best_endpoints = (endpoint1, endpoint2)

    if best_endpoints is None:
        print(base_min)
        print(base_max)
        print("measure_vertical_segment: No valid segment found.")
        return None, None

    # Designate the point with the smaller y-value as the "top" and the other as the "bottom."
    pt1, pt2 = best_endpoints
    if pt1[1] < pt2[1]:
        top_point = pt1
        bottom_point = pt2
    else:
        top_point = pt2
        bottom_point = pt1

    return bottom_point, top_point  # (x_bottom, y_bottom), (x_top, y_top)

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

def measure_ruler(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 200, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=1500, maxLineGap=15)
        pixel_count = 0
        drawn_x1 = drawn_y1 = drawn_x2 = drawn_y2 = 0  # initialize the coordinates of the line

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y1 - y2) > 300:
                    if abs(y1 - y2) > pixel_count:
                        pixel_count = abs(y1 - y2)
                        drawn_x1, drawn_y1, drawn_x2, drawn_y2 = x1, y1, x2, y2

        delta_y = drawn_y2 - drawn_y1
        delta_x = drawn_x2 - drawn_x1
        angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi
        if angle < -45:
            angle += 90
        center = (image.shape[1] // 2, image.shape[0] // 2)
        
        # Rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

        print(f"Default line length: {abs(drawn_y2 - drawn_y1)}")

        points = np.array([[drawn_x1, drawn_y1], [drawn_x2, drawn_y2]], dtype=np.float32)
        rotated_points = cv2.transform(np.array([points]), rotation_matrix)[0]
        rotated_x1, rotated_y1 = rotated_points[0]
        rotated_x2, rotated_y2 = rotated_points[1]
        print(f"Adjusted line length: {abs(int(rotated_y1) - int(rotated_y2))}")
        cv2.line(rotated_image, (int(rotated_x1), int(rotated_y1)), (int(rotated_x2), int(rotated_y2)), (0, 0, 255), 2)
        # Save the images
        pixel_count = abs(int(rotated_y1)-int(rotated_y2))
        cv2.imwrite(f"{image}_lines.jpg", rotated_image)
        if abs(int(rotated_y1)-int(rotated_y2)):
            conversion_factor = 10/(pixel_count/15.6)
            return conversion_factor
    except:
        print("Error in ruler measurement using default conversion")
        return