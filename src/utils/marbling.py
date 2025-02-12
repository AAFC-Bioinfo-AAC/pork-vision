import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

def extract_muscle_region(rotated_image, muscle_mask):
    """
    Extracts the muscle portion of the image using the provided mask.
    """
    muscle_region = cv2.bitwise_and(rotated_image, rotated_image, mask=muscle_mask)
    return muscle_region

def enhance_contrast_and_sharpen(image):
    """
    Applies contrast enhancement and sharpening.
    Converts the image to grayscale, applies CLAHE, sharpens it, and
    then converts back to a 3-channel image for compatibility.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Apply sharpening filter
    sharpen_kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    
    # Convert back to 3-channel image (for compatibility with later OpenCV functions)
    sharpened_3ch = cv2.merge([sharpened, sharpened, sharpened])
    
    return sharpened_3ch

def gaussian_threshold(image):
    """
    Uses Gaussian adaptive thresholding to generate a binary image.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=0.1, tileGridSize=(8, 8))
    gray_image = clahe.apply(gray_image)
    binary_image = cv2.adaptiveThreshold(
        gray_image, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 81, -36
    )
    binary_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    return binary_image

def overlay_images(image_1, image_2):
    """
    Overlays two images using weighted addition.
    """
    overlay_image = cv2.addWeighted(image_1, 1.0, image_2, 1.0, 0)
    return overlay_image

def extract_marbling(enhanced_image, muscle_mask):
    """
    Extracts marbling by thresholding the brightest pixels in the enhanced image.
    The thresholding uses a Gaussian adaptive method and then restricts the output
    to only those pixels corresponding to the muscle region.
    """
    # Use Gaussian adaptive thresholding with updated parameters for higher resolution
    binary = gaussian_threshold(enhanced_image)  # Returns a BGR binary image
    binary_gray = cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY)  # Convert to single-channel
    
    # Restrict thresholding to muscle pixels (mask out background)
    marbling_mask = cv2.bitwise_and(binary_gray, muscle_mask)
    
    # Reduce the amount of blurring and morphological processing to preserve details
    marbling_mask = cv2.GaussianBlur(marbling_mask, (1, 1), 0)
    kernel = np.ones((1, 1), np.uint8)
    marbling_mask = cv2.morphologyEx(marbling_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    marbling_mak = contour_detection(marbling_mask)
    return marbling_mask

def contour_detection(marbling_mask):
    contours, hierarchy = cv2.findContours(marbling_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_mask = cv2.drawContours(marbling_mask, contours, 0, color=(0,0,0), thickness=30)
    return contour_mask

def convert_fat_color(image):
    '''
    Converts pure white in the images, into a yellow.
    Input: Matlike image.
    Output: Image where only the white has been shifted to yellow
    '''
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    white_mask = (lab_image[:,:,0] == 255) & (lab_image[:,:, 1] == 128) & (lab_image[:,:,2] == 128) ##Checks for pure white.
    lab_image[white_mask, 2] = lab_image[white_mask, 2] + 100
    lab_image = np.clip(lab_image, 0, 255)
    lab_final = cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return lab_final

def process_marbling(rotated_image, muscle_mask, output_dir="output/marbling", base_filename=None):
    """
    Full pipeline to extract marbling from the rotated muscle image.
    This function processes the given image and muscle mask (both as arrays),
    and saves intermediate images (with descriptive filenames) to the output directory.
    
    Parameters:
      rotated_image: The rotated image (as a NumPy array).
      muscle_mask: The binary muscle mask (as a NumPy array).
      output_dir: Directory where output images will be saved.
      base_filename: Optional base name for saving files. If not provided, defaults to 'marbling_result'.
    
    Returns:
      marbling_mask: The final marbling mask (as a single-channel image).
    """
    if base_filename is None:
        base_filename = "marbling_result"
    
    # Extract the muscle region from the image using the provided mask
    muscle_region = extract_muscle_region(rotated_image, muscle_mask)
    
    # Enhance contrast and sharpen the muscle region
    enhanced_image = enhance_contrast_and_sharpen(muscle_region)
    
    # Extract the marbling mask, considering only muscle pixels
    marbling_mask = extract_marbling(enhanced_image, muscle_mask)
    
    # Create an overlay image to visualize marbling on the original image
    overlay = overlay_images(rotated_image, cv2.cvtColor(marbling_mask, cv2.COLOR_GRAY2BGR))
    overlay = convert_fat_color(overlay)
    
    # Create a subfolder for the current image (named by the base filename)
    base_output_dir = os.path.join(output_dir, base_filename)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Save each intermediate image inside the subfolder
    muscle_region_filename = os.path.join(base_output_dir, f"{base_filename}_muscle_region.jpg")
    enhanced_filename      = os.path.join(base_output_dir, f"{base_filename}_enhanced.jpg")
    marbling_mask_filename = os.path.join(base_output_dir, f"{base_filename}_marbling_mask.jpg")
    overlay_filename       = os.path.join(base_output_dir, f"{base_filename}_overlay.jpg")
    
    cv2.imwrite(muscle_region_filename, muscle_region)
    cv2.imwrite(enhanced_filename, enhanced_image)
    cv2.imwrite(marbling_mask_filename, marbling_mask)
    cv2.imwrite(overlay_filename, overlay)
    
    return marbling_mask

####################################
# --- MAIN FUNCTION FOR TESTING ---
####################################
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python marbling.py <rotated_image_path> <muscle_mask_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    muscle_mask_path = sys.argv[2]
    
    rotated_image = cv2.imread(image_path)
    if rotated_image is None:
        print(f"Error: Unable to read image from {image_path}")
        sys.exit(1)
    
    muscle_mask = cv2.imread(muscle_mask_path, cv2.IMREAD_GRAYSCALE)
    if muscle_mask is None:
        print(f"Error: Unable to read muscle mask from {muscle_mask_path}")
        sys.exit(1)
    
    # Derive a base filename from the input image path for saving results
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    
    process_marbling(rotated_image, muscle_mask, base_filename=base_filename)