import cv2
import numpy as np
import pandas as pd
import os
from skimage.exposure import match_histograms


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

canadian_rgb_standard = np.array([
    (175, 83, 81),   # C6
    (185, 100, 87),  # C5
    (190, 114, 97),  # C4
    (194, 129, 106), # C3
    (203, 146, 117), # C2
    (208, 160, 125), # C1
    (217, 172, 131)  # C0
], dtype=np.float32)

japanese_rgb = np.array([
    (146, 46, 44),   # J6
    (153, 65, 55),   # J5
    (168, 85, 67),   # J4
    (178, 103, 80),   # J3
    (193, 126, 97),  # J2
    (199, 144, 105)   # J1
], dtype=np.float32)

japanese_rgb_standard = np.array([
    (124, 33, 26),  # J6
    (144, 40, 36),  # J5
    (160, 72, 50),  # J4
    (173, 88, 60),  # J3
    (188, 110, 70), # J2
    (191, 124, 75), # J1
], dtype=np.float32)

#######################################
#####Standardization Functions#########
#######################################
def white_balance(image, option):
    '''
    Balance the white in an image
    Helps reduce lighting impact.
    Simple WB: Result is closer to the original, but increased variance.
    Learning WB: Result has less variance.
    '''
    if option == "SimpleWB":
        result = cv2.xphoto.createSimpleWB().balanceWhite(image)
    if option == "LearnWB":
        result = cv2.xphoto.createLearningBasedWB().balanceWhite(image)
    return result

def LAB_check(reference_image, image):
    lab_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)
    lab_current = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ref, a_ref, b_ref = cv2.split(lab_ref)
    l_cur, a_cur, b_cur = cv2.split(lab_current)
    if (np.mean(l_cur)/np.mean(l_ref) < 0.99) or (np.mean(l_cur)/np.mean(l_ref) > 1.01) \
    or (np.mean(a_cur)/np.mean(a_ref) < 0.99) or (np.mean(a_cur)/np.mean(a_ref) > 1.01) \
    or (np.mean(b_cur)/np.mean(b_ref) < 0.99) or (np.mean(b_cur)/np.mean(b_ref) > 1.01):
        #print("Outside margin of error, add to correction list")
        standard_img  = reference_standardize(image, reference_image)
        return standard_img
    else:
        #print("Within margin of error, no need to standardize")
        return image

def reference_standardize(image, reference_image):
    '''
    Takes in a list of images, and a reference.
    Standardizes the list of images to the reference
    by matching histograms.
    Returns standardized image.
    '''
    standard_img = match_histograms(image, reference_image, channel_axis=-1)
    #standard_img = cv2.medianBlur(standard_img, 3) Used just to approximate Category cutoffs
    return standard_img

def execute_color_standardization(image):
    reference_image = cv2.imread('data/reference_images/2704_LdLeanColor.JPG')
    #reference_image = white_balance(reference_image, "SimpleWB")

    #balance = white_balance(image, "SimpleWB")
    standardized_image = LAB_check(reference_image, image)
    return standardized_image


############################
#####ANALYSIS FUNCTIONS#####
############################
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

def colour_grading(image, muscle_mask, marbling_mask, output_dir, image_id):
    """Performs color grading on the lean muscle area (excluding marbling) and saves results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gets the lean mask (muscle area excluding marbling)
    lean_mask = cv2.subtract(muscle_mask, marbling_mask)
    
    # Performs vectorized color analysis for Canadian and Japanese standards
    canadian_classified = classify_rgb_vectorized(image, canadian_rgb, lean_mask)
    japanese_classified = classify_rgb_vectorized(image, japanese_rgb, lean_mask)
    
    # Applies LUTs for visualization with a black background
    canadian_lut_image = apply_lut(canadian_classified, list(range(7)), canadian_rgb, lean_mask)
    japanese_lut_image = apply_lut(japanese_classified, list(range(6)), japanese_rgb, lean_mask)
    
    # Creates a standardization for the image
    standard_img = execute_color_standardization(image)

    # Save results
    base_output_dir = os.path.join(output_dir, image_id)
    os.makedirs(base_output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_canadian_lut.png"), canadian_lut_image)
    cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_japanese_lut.png"), japanese_lut_image)
    cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_STANDARDIZED.png"), standard_img)

    return canadian_classified, japanese_classified, lean_mask

def save_colouring_csv(id_list, canadian_classified_list, japanese_classified_list, lean_mask_list, output_csv_path):
    """Save the color analysis results for multiple images to a CSV file, ensuring all standards are represented."""

    def generate_pixel_stats(classified_image, lean_mask, total_pixels, standard_type, image_id):
        """Generate pixel counts and percentages for each standard, ensuring all standards are represented."""
        if standard_type == "Cdn":
            standards = [f"CdnStd{i}" for i in range(7)]  # Canadian standards: CdnStd0 to CdnStd6
        else:
            standards = [f"JpnStd{i + 1}" for i in range(6)]  # Japanese standards: JpnStd1 to JpnStd6

        # Count pixels for each classification
        unique, counts = np.unique(classified_image[lean_mask > 0], return_counts=True)
        pixel_counts = dict(zip(unique, counts))  # Maps unique classifications to their counts
        
        # Generate data for all standards, filling missing ones with 0
        data = []
        for i, standard in enumerate(standards):
            count = pixel_counts.get(i, 0)
            percentage = round((count / total_pixels) * 100, 2) if total_pixels > 0 else 0.00
            data.append({
                "image_id": image_id,
                "standard": standard,
                "pixel_count": count,
                "percentage": percentage
            })
        return data

    all_data = []
    
    for image_id, canadian_classified, japanese_classified, lean_mask in zip(id_list, canadian_classified_list, japanese_classified_list, lean_mask_list):
        total_pixels = np.count_nonzero(lean_mask)  # Total number of lean pixels

        # Calculate statistics for Canadian and Japanese standards
        canadian_stats = generate_pixel_stats(canadian_classified, lean_mask, total_pixels, "Cdn", image_id)
        japanese_stats = generate_pixel_stats(japanese_classified, lean_mask, total_pixels, "Jpn", image_id)
        
        # Add both sets of statistics to the final data list
        all_data.extend(canadian_stats + japanese_stats)

    # Convert all data to a DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV (overwrite each time)
    df.to_csv(output_csv_path, index=False)