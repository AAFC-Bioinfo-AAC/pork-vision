from utils.imports import *

def dilate_mask(binary_mask, kernel_size=15):
    """
    Dilates the given binary mask to create a 'band' around the object.

    Parameters:
    - binary_mask (np.ndarray): A binary (0/255) mask.
    - kernel_size (int): Size of the dilation kernel.

    Returns:
    - np.ndarray: Dilated mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(binary_mask, kernel, iterations=1)

def find_largest_contour(mask, min_area=500):
    """
    Finds the largest contour in a binary mask, ignoring contours smaller than min_area.

    Parameters:
    - mask (np.ndarray): Binary mask where contours will be found.
    - min_area (int): Minimum area to be considered valid.

    Returns:
    - tuple: (x, y, w, h) bounding box of the largest valid contour, or None if none found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid_contours:
        return None
    largest_c = max(valid_contours, key=cv2.contourArea)
    return cv2.boundingRect(largest_c)

def preprocess_mask(binary_mask):
    """
    Smoothes small holes/gaps in the binary mask.

    Parameters:
    - binary_mask (np.ndarray): A binary mask of the object.

    Returns:
    - np.ndarray: Smoothed binary mask.
    """
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

def rotate_image(image, angle):
    """
    Rotates an image about its center by a given angle in degrees.

    Parameters:
    - image (np.ndarray): The input image or mask.
    - angle (float): Rotation angle in degrees.

    Returns:
    - np.ndarray: Rotated image or mask.
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    abs_cos = abs(rot_matrix[0, 0])
    abs_sin = abs(rot_matrix[0, 1])

    # Compute the new dimensions of the rotated image
    new_w = int(h * abs_sin + w * abs_cos)
    new_h = int(h * abs_cos + w * abs_sin)
    
    # Adjust the rotation matrix to account for translation (no cropping)
    rot_matrix[0, 2] += (new_w / 2) - center[0]
    rot_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Perform the rotation with the new dimensions
    rotated_image = cv2.warpAffine(image, rot_matrix, (new_w, new_h))
    return rotated_image

def isolate_adjacent_fat(muscle_mask, fat_mask, debug_messages, dilation_size=15, min_area=500):
    """
    Extracts only the portion of fat that lies adjacent (within 'dilation_size' pixels) to the muscle.

    Steps:
    1. Dilate the muscle mask to form a band around the muscle.
    2. Intersect this band with the fat mask -> 'adjacent_fat'.
    3. Morphologically close the result to remove small holes.
    4. Keep the largest contour (above min_area).

    Parameters:
    - muscle_mask (np.ndarray): Binary mask of the muscle.
    - fat_mask (np.ndarray): Binary mask of the fat.
    - dilation_size (int): Pixel size for dilation to define adjacency.
    - min_area (int): Minimum area for a valid fat region.

    Returns:
    - tuple: (x, y, w, h) bounding box of the adjacent fat region, or None if none found.
    """
    debug_messages.append(f"dilation size: {dilation_size}, minimum valid area: {min_area}")
    # 1. Dilate the muscle mask
    dilated_muscle = dilate_mask(muscle_mask, kernel_size=dilation_size)

    # 2. Intersection with the fat mask
    adjacent_fat = cv2.bitwise_and(dilated_muscle, fat_mask)

    # 3. Smooth the adjacent fat region
    adjacent_fat = preprocess_mask(adjacent_fat)
    # 4. Find largest valid contour
    return find_largest_contour(adjacent_fat, min_area=min_area), debug_messages

def initial_orientation_correction(original_image, muscle_mask, fat_mask, debug_messages, rotation=cv2.ROTATE_90_CLOCKWISE):
    '''
    Corrects the initial orientation (if the image is upside down or sideways relative to the fat up).
    Input: Original_image, muscle_mask, fat_mask.
    Output: Rotated_image, muscle_mask, fat_mask
    '''
    height, width, _ = original_image.shape
    if width<height: # If the image is vertical.
        debug_messages.append("Image is vertical, correcting")
        debug_messages.append(f"Height before rotation = {height}, width before rotation = {width}")
        rotated_image = cv2.rotate(original_image, rotation)
        rotated_muscle_mask = cv2.rotate(muscle_mask, rotation)
        rotated_fat_mask = cv2.rotate(fat_mask, rotation)
        height, width, _ = rotated_image.shape
        debug_messages.append(f"Height after rotation {height}, width after rotation {width}")
    else:
        rotated_image = original_image
        rotated_muscle_mask = muscle_mask
        rotated_fat_mask = fat_mask
    fat_pixels = np.where(rotated_fat_mask == 255)
    muscle_pixels = np.where(rotated_muscle_mask == 255)


    # Get the maximum y value for the muscle mask
    muscley_value = np.max(muscle_pixels[0])

    # Get the x values where y = muscley_value
    musclex_values = muscle_pixels[1][muscle_pixels[0] == muscley_value]

    # If multiple x values exist at this y, choose the first one:
    musclex_value = musclex_values[0]

    # Get the y values where x = musclex_value in fat mask
    faty_values_at_musclex = fat_pixels[0][fat_pixels[1] == musclex_value]

    # Get the maximum y value at this x position in the fat mask
    faty_value = np.max(faty_values_at_musclex)
    #If the fat is below the muscle rotate to fix.
    if faty_value > muscley_value:
        debug_messages.append(f"CORRECTING: Bottom fat detected at {faty_value}, while the bottom muscle is {muscley_value}, so fat is below muscle.")
        if width>height:
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)
            rotated_muscle_mask = cv2.rotate(rotated_muscle_mask, cv2.ROTATE_180)
            rotated_fat_mask = cv2.rotate(rotated_fat_mask, cv2.ROTATE_180)
            return rotated_image,rotated_muscle_mask,rotated_fat_mask,debug_messages

    debug_messages.append(f"GOOD: Fat detected at {faty_value} while muscle is at {muscley_value} so fat above muscle")
    return rotated_image, rotated_muscle_mask, rotated_fat_mask, debug_messages
    
def orient_muscle_and_fat_using_adjacency(original_image, muscle_mask, fat_mask, outlier, debug_messages):
    """
    Orients the image so that the thin strip of fat adjacent to the muscle is on top.

    Steps:
    1. Get bounding boxes for muscle and the adjacent fat region.
    2. Compare bounding box centers to decide 0°, +90°, -90°, or 180° rotation.

    Parameters:
    - original_image (np.ndarray): Original image.
    - muscle_mask (np.ndarray): Binary mask of the muscle region.
    - fat_mask (np.ndarray): Binary mask of the fat region.

    Returns:
    - rotated_image (np.ndarray): Rotated image.
    - rotated_muscle_mask (np.ndarray): Rotated muscle mask.
    - rotated_fat_mask (np.ndarray): Rotated fat mask.
    - final_angle (int): The rotation angle in degrees (0, 90, -90, or 180).
    """

    # 1. Find bounding box of muscle
    muscle_contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not muscle_contours:
        outlier = "Y"
        debug_messages.append("No muscle found. Skipping orientation.")
        return original_image, muscle_mask, fat_mask, 0, outlier, debug_messages
    rotated_image, rotated_muscle_mask, rotated_fat_mask, debug_messages = initial_orientation_correction(original_image, muscle_mask, fat_mask, debug_messages)
    height,width, _ = rotated_image.shape
    if width<height:
        outlier = "Y"
        debug_messages.append("ERROR: Image is vertical after rotation.")
    return rotated_image, rotated_muscle_mask, rotated_fat_mask, 0, outlier, debug_messages