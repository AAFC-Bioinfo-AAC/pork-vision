from utils.imports import *



# =============================================================================
# Helper Functions
# =============================================================================
def extract_muscle_region(rotated_image, muscle_mask):
    """
    Extracts the muscle portion of the image using the provided mask.
    Creates a kernel and fits the perimeter of the muscle mask onto it.
    Finds the light areas in the perimeter (likely fat going into the mask).
    Removes the perimeter fat and blood.
    Returns a muscle mask with the interfering regions removed.
    """
    kernel = np.ones((1,1), np.uint8)
    perimeter = cv2.morphologyEx(muscle_mask, cv2.MORPH_GRADIENT, kernel)
    gray = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)
    _, light_areas = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)

    target_areas = cv2.bitwise_and(perimeter, light_areas)
    
    selective_mask = np.where(target_areas == 255, 0, muscle_mask)
    muscle_region = cv2.bitwise_and(rotated_image, rotated_image, mask=selective_mask).astype(np.uint8)

    upper_red = np.array([100,100, 255], dtype=np.uint8)
    lower_red = np.array([0,0,50], dtype=np.uint8)
    blood_mask = cv2.inRange(muscle_region, lower_red, upper_red)
    perimeter = cv2.morphologyEx(selective_mask, cv2.MORPH_GRADIENT, kernel)

    target_areas = cv2.bitwise_and(perimeter, blood_mask)
    selective_mask = np.where(target_areas == 255, 0, selective_mask)
    muscle_region = cv2.bitwise_and(rotated_image, rotated_image, mask=selective_mask).astype(np.uint8)




    return muscle_region, selective_mask

def filter_muscle_region(muscle_region, muscle_mask, canadian_standards):
  '''
  Using the Canadian standards color chart, we remove any areas that were mistakenly
  left over (more likely fat than muscle).
  Returns filtered muscle region.
  '''

  std_0 = canadian_standards[-1]
  std_0 -= 10
  print(std_0)
  kernel = np.ones((1,1), np.uint8)
  perimeter = cv2.morphologyEx(muscle_mask, cv2.MORPH_GRADIENT, kernel)
  perimeter_inv = cv2.bitwise_not(perimeter)
  muscle_region = cv2.cvtColor(muscle_region, cv2.COLOR_BGR2RGB)
  lower_white = np.array(std_0, dtype=np.uint8)
  upper_white = np.array([255,255,255], dtype=np.uint8)
  white_mask = cv2.inRange(muscle_region, lower_white, upper_white).astype(np.uint8)
  target_areas = cv2.bitwise_and(perimeter_inv,white_mask)
  muscle_region[target_areas == 255] = [0, 0, 0]
  return muscle_region

def clahe_contrast_enhancement(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    '''
    Apply CLAHE in order to better increase the accuracy of results. 
    '''
    # Image is single-channel (grayscale), apply CLAHE directly
    if len(image.shape) == 2:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_image = clahe.apply(image)
    else:  # Color image
        # Convert to LAB color space to enhance the L channel (luminance)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        
        # Merge the CLAHE-enhanced L channel back with the A and B channels
        enhanced_lab = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
    return enhanced_image

def background_subtraction(image, kernel_size=11):
    """
    Performs background subtraction using a Gaussian blur.
    If the input is a color image, it is first converted to grayscale.
    
    Parameters:
      image: Input image (grayscale or color).
      kernel_size: Size of the Gaussian kernel (must be odd).
      
    Returns:
      subtracted: The background-subtracted image.
    """
    if len(image.shape) == 3:
        enhanced_image = clahe_contrast_enhancement(image)
        gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    bg = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    subtracted = cv2.subtract(gray, bg)
    subtracted = contour_detection(subtracted)
    return subtracted

def dynamic_contrast_stretch(image):
    """
    Stretches the contrast of the input 8-bit grayscale image dynamically
    using normalization to map the intensity range to 0-255.
    
    Parameters:
      image: Input 8-bit image (grayscale).
      
    Returns:
      stretched: The contrast-stretched image.
    """
    stretched = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return np.uint8(stretched)

def apply_custom_lut(image, lut=cv2.COLORMAP_JET):
    """
    Applies a custom Look-Up Table (LUT) for pseudo‑colour imaging.
    
    Parameters:
      image: Input 8-bit single‑channel image.
      lut: OpenCV colormap to use (default is COLORMAP_JET).
      
    Returns:
      pseudo: The pseudo‑colour image.
    """
    pseudo = cv2.applyColorMap(image, lut)
    return pseudo

def perform_preprocessing(image, kernel_size=11, lut=cv2.COLORMAP_JET):
    """
    Performs enhanced preprocessing that includes background subtraction,
    dynamic contrast stretching, and applying a custom LUT for pseudo‑colour imaging.
    
    Parameters:
      image: Input image (color or grayscale).
      kernel_size: Kernel size for Gaussian background subtraction.
      lut: LUT to be applied.
      
    Returns:
      contrast_stretched: The contrast‑stretched (grayscale) image.
      pseudo_color: The pseudo‑colour version of the preprocessed image.
    """
    bg_subtracted = background_subtraction(image, kernel_size=kernel_size)
    contrast_stretched = dynamic_contrast_stretch(bg_subtracted)
    pseudo_color = apply_custom_lut(contrast_stretched, lut=lut)
    return contrast_stretched, pseudo_color

def contour_detection(marbling_mask):
    """ Removes the contour line from the marbling mask. """
    contours, _ = cv2.findContours(marbling_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(marbling_mask, contours, -1, color=0, thickness=6)  # Fill the contour with black
    return marbling_mask

def smooth_marbling_mask(marbling_mask, kernel_size=(5, 5)):
    """
    Applies Gaussian blur to the marbling mask for smoothing pixel boundaries.

    Parameters:
      marbling_mask: Binary marbling mask (0/255) as a single-channel 8-bit image.
      kernel_size: Size of the Gaussian kernel (default is (5, 5)).

    Returns:
      smoothed_mask: The smoothed marbling mask.
    """
    smoothed_mask = cv2.GaussianBlur(marbling_mask, kernel_size, 0)
    _, binary_mask = cv2.threshold(smoothed_mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask

def contrast_enhancement(image, gamma=0.3):
    """
    Applies gamma correction to exaggerate brightness differences in the input image.
    
    Parameters:
      image: Input 8-bit image (should be a single-channel image).
      gamma: Gamma correction factor (default is 0.3).
      
    Returns:
      enhanced: Image after gamma correction.
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8")
    enhanced = cv2.LUT(image, table)
    return enhanced

# =============================================================================
# Marbling Analysis Functions
# =============================================================================
def particle_analysis(binary_mask, min_area=5):
    """
    Performs particle analysis on a binary mask using connected component analysis.
    Filters out components smaller than a specified minimum area.
    
    Parameters:
      binary_mask: Binary image (0 and 255 values) as a single‑channel 8-bit image.
      min_area: Minimum area (in pixels) for a component to be considered.
      
    Returns:
      refined_mask: Binary mask containing only the components above the area threshold.
      total_area: Sum of areas for all accepted components.
    """
    if len(binary_mask.shape) == 3:
        mask = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    else:
        mask = binary_mask.copy()
    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    refined_mask = np.zeros_like(thresh)
    total_area = 0
    for i in range(1, num_labels):  # skip background label 0
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            refined_mask[labels == i] = 255
            total_area += area
    return refined_mask, total_area

def calculate_marbling_percentage(marbling_mask, muscle_mask):
    """
    Calculates the marbling percentage relative to the inner area of the muscle mask,
    excluding the outer contour of the muscle.

    Parameters:
      marbling_mask: Binary mask (0/255) for marbling.
      muscle_mask: Binary mask (0/255) for muscle (lean) area.

    Returns:
      percentage: Marbling area as a percentage of the inner muscle area (excluding the contour).
    """
    # Find contours of the muscle mask
    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return 0.0  # Return 0 if no contours found

    # Create a new mask with only the inner area of the muscle (exclude the contour)
    inner_muscle_mask = np.zeros_like(muscle_mask)
    cv2.drawContours(inner_muscle_mask, contours, -1, 255, thickness=cv2.FILLED)

    # Ensure the marbling mask is restricted to the inner muscle region
    inner_marbling_mask = cv2.bitwise_and(marbling_mask, inner_muscle_mask)
    
    # Calculate areas
    marbling_area = cv2.countNonZero(inner_marbling_mask)
    muscle_area = cv2.countNonZero(inner_muscle_mask)
    
    if muscle_area == 0:
        return 0.0
    
    percentage = (marbling_area / muscle_area) * 100.0
    return percentage, muscle_area

# =============================================================================
# Full Marbling Processing Pipeline
# =============================================================================
def process_marbling(rotated_image, muscle_mask, output_dir, canadian_standards, minimal, base_filename=None):
    """
    Full pipeline to extract marbling from the rotated muscle image.
    This pipeline uses the pseudo‑colour image (from COLORMAP_JET) and extracts its blue channel,
    applies extreme contrast enhancement (gamma correction) within the muscle mask,
    and then thresholds to capture every pixel of fat.
    
    Parameters:
      rotated_image: The rotated image (as a NumPy array).
      muscle_mask: The binary muscle mask (as a NumPy array).
      output_dir: Directory where output images will be saved.
      base_filename: Optional base name for saving files; defaults to 'marbling_result'.
    
    Returns:
      refined_marbling_mask: Final refined marbling mask (a single‑channel image).
      marbling_percentage: Marbling area as a percentage of the muscle area.
    """
    # Extract the muscle region
    muscle_region, selective_mask= extract_muscle_region(rotated_image, muscle_mask)

    # Obtain the pseudo‑colour image from enhanced preprocessing
    _, pseudo_color = perform_preprocessing(muscle_region, kernel_size=11, lut=cv2.COLORMAP_JET)
    
    # Instead of converting the full pseudo‑colour to grayscale, extract the blue channel.
    # In OpenCV's BGR, blue is at index 0.
    pseudo_blue = pseudo_color[:, :, 0]
    
    # Apply contrast enhancement on the blue channel to boost fat brightness.
    enhanced_blue = contrast_enhancement(pseudo_blue, gamma=0.4)
    
    # Use a low fixed threshold on the enhanced blue channel so that every bright fat pixel is captured.
    ret, thresh_blue = cv2.threshold(enhanced_blue, 70, 255, cv2.THRESH_BINARY)
    refined_marbling_mask = smooth_marbling_mask(thresh_blue, kernel_size=(7, 7))
    # Refine the marbling mask
    refined_marbling_mask, _ = particle_analysis(refined_marbling_mask, min_area=60)
    muscle_region = filter_muscle_region(muscle_region, selective_mask, canadian_standards)
    gray_muscle_region = cv2.cvtColor(muscle_region, cv2.COLOR_BGR2GRAY)
    condition_mask = (selective_mask == 255) & (gray_muscle_region == 0)
    refined_marbling_mask[condition_mask] = 255

    # Calculate marbling percentage relative to the muscle area
    marbling_percentage, area_px = calculate_marbling_percentage(refined_marbling_mask, selective_mask)
    
    # Save images in a subfolder
    if minimal == False:
      if base_filename is None:
        base_filename = "marbling_result"
      base_output_dir = os.path.join(output_dir, base_filename)
      os.makedirs(base_output_dir, exist_ok=True)
      cv2.imwrite(os.path.join(base_output_dir, f"{base_filename}_marbling_mask.jpg"), refined_marbling_mask)
      cv2.imwrite(os.path.join(base_output_dir, f"{base_filename}_selective_muscle_mask.jpg"), selective_mask)
      cv2.imwrite(os.path.join(base_output_dir, f"{base_filename}_original_muscle_mask.jpg"), muscle_mask)

    return refined_marbling_mask, selective_mask, marbling_percentage, area_px,

# ==============================
# Saving results
# ==============================
def save_marbling_csv(id_list, fat_percentage_list, output_csv_path):
    df = pd.DataFrame({
        "image_id" : id_list,
        "fat_percentage" : fat_percentage_list
    })

    df.to_csv(output_csv_path, index=False)