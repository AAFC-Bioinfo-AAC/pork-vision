import cv2
import numpy as np
import os

# RGB values for Canadian and Japanese lean color standards
canadian_rgb = np.array([
    (170, 87, 95),   # C6
    (177, 101, 103), # C5
    (195, 125, 125), # C4
    (204, 146, 142), # C3
    (209, 162, 152), # C2
    (211, 175, 161), # C1
    (215, 184, 164)  # C0
], dtype=np.float32)

japanese_rgb = np.array([
    (146, 46, 44),   # J6
    (153, 65, 55),   # J5
    (168, 85, 67),   # J4
    (178, 103, 80),   # J3
    (193, 126, 97),  # J2
    (199, 144, 105)   # J1
], dtype=np.float32)

def classify_rgb_vectorized(image, standards, lean_mask):
    """Vectorized classification of RGB pixels using Euclidean distance."""
    h, w, _ = image.shape
    classified_image = np.zeros((h, w), dtype=np.uint8)

    # Apply the lean mask to restrict analysis to lean muscle pixels
    lean_pixels = image[lean_mask > 0].astype(np.float32)  # Extract lean muscle pixels as a (N, 3) array

    # Calculate Euclidean distances to each standard for all lean pixels
    distances = np.linalg.norm(lean_pixels[:, None] - standards[None, :], axis=2)  # (N, num_standards)

    # Find the index of the closest standard for each pixel
    closest_standard_indices = np.argmin(distances, axis=1)  # (N,)

    # Map the classified indices back to the original image shape
    classified_image[lean_mask > 0] = closest_standard_indices

    return classified_image

def apply_lut(image, category_values, lut_values, mask):
    """Applies a custom LUT to the classified image and ensures the background is black."""
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(lut_values):
        lut[category_values[i]] = [b, g, r]

    colored_image = cv2.LUT(cv2.merge([image] * 3), lut)

    # Ensure the background is black
    colored_image[mask == 0] = [0, 0, 0]

    return colored_image

def color_grading(image, muscle_mask, marbling_mask, output_dir, image_id):
    """Performs color grading on the lean muscle area (excluding marbling) and saves results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensures the image is in RGB format
    if len(image.shape) < 3 or image.shape[2] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Gets the lean mask (muscle area excluding marbling)
    lean_mask = cv2.subtract(muscle_mask, marbling_mask)
    
    # Performs vectorized color analysis for Canadian and Japanese standards
    canadian_classified = classify_rgb_vectorized(image, canadian_rgb, lean_mask)
    japanese_classified = classify_rgb_vectorized(image, japanese_rgb, lean_mask)
    
    # Applies LUTs for visualization with a black background
    canadian_lut_image = apply_lut(canadian_classified, list(range(7)), canadian_rgb, lean_mask)
    japanese_lut_image = apply_lut(japanese_classified, list(range(6)), japanese_rgb, lean_mask)
    
    # Save results
    base_output_dir = os.path.join(output_dir, image_id)
    os.makedirs(base_output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_canadian_lut.png"), canadian_lut_image)
    cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_japanese_lut.png"), japanese_lut_image)