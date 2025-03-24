from utils.imports import *

# RGB values for Canadian lean color standards
canadian_rgb_standard = np.array([
    ((222+218+219+217+217+213+216+220+217+219)/10, (191+186+188+186+185+182+185+189+186+188)/10, (173+168+169+167+166+164+166+171+167+170)/10), # C0
    ((217+211+212+210+210+206+210+214+211+213)/10, (178+173+174+172+172+168+172+176+173+175)/10, (166+160+160+159+159+154+157+164+160+164)/10), # C1
    ((214+207+208+207+206+202+206+212+207+210)/10, (166+160+161+159+159+155+159+164+159+163)/10, (156+150+151+149+149+145+149+154+149+153)/10), # C2
    ((205+198+198+196+197+193+197+202+197+201)/10, (148+141+142+139+140+136+140+146+140+145)/10, (142+135+136+133+135+129+134+141+135+139)/10), # C3
    ((197+192+192+190+190+186+191+194+191+194)/10, (129+123+124+121+121+117+122+127+122+126)/10, (128+122+123+121+121+117+121+127+122+126)/10), # C4
    ((190+184+183+182+182+177+184+186+184+185)/10, (110+103+103+101+101+98+103+106+103+106)/10, (115+109+109+107+108+103+108+113+108+111)/10), # C5
    ((179+172+171+169+170+166+172+175+172+174)/10, (96+90+90+87+88+85+89+93+89+92)/10, (103+98+98+96+96+93+97+102+96+101)/10),   # C6
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

    def match_channel(source, target):
        # Calculate the mean and standard deviation of source and target channels
        mean_source, std_source = cv2.meanStdDev(source)
        mean_target, std_target = cv2.meanStdDev(target)
    
        # Normalize the source channel
        normalized_source = (source - mean_source) / std_source
        matched_channel = (normalized_source * std_target) + mean_target
    
        # Clip the values to be in valid LAB range [0, 255]
        matched_channel = np.clip(matched_channel, 0, 255).astype(np.uint8)
        return matched_channel
    lab_source = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_target = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

    L_source, A_source, B_source = cv2.split(lab_source)
    L_target, A_target, B_target = cv2.split(lab_target)
    L_matched = match_channel(L_source, L_target)
    A_matched = match_channel(A_source, A_target)
    B_matched = match_channel(B_source, B_target)
    lab_matched = cv2.merge([L_matched, A_matched, B_matched])
    standard_img = cv2.cvtColor(lab_matched, cv2.COLOR_LAB2BGR)

    #standard_img = match_histograms(image, reference_image, channel_axis=-1)
    #standard_img = cv2.medianBlur(standard_img, 3) #Used just to approximate Category cutoffs
    return standard_img

def execute_color_standardization(image, reference_path):
    reference_image = cv2.imread(reference_path)
    #reference_image = white_balance(reference_image, "SimpleWB")

    #balance = white_balance(image, "SimpleWB")
    standardized_image = LAB_check(reference_image, image)
    return standardized_image


############################
#####ANALYSIS FUNCTIONS#####
############################
def classify_rgb_vectorized(image, standards, lean_mask):
    """Vectorized classification of RGB pixels using Euclidean distance."""
    # Convert the image to RGB (if it's in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape the image to a (h*w, 3) matrix of RGB values
    h, w, _ = image.shape
    image_rgb_reshaped = image_rgb.reshape(-1, 3)

    # Create a mask to filter out pixels where lean_mask == 0
    mask = lean_mask.reshape(-1) > 0

    # Compute the distances between each pixel and each standard
    distances = np.linalg.norm(image_rgb_reshaped[mask][:, np.newaxis] - standards, axis=2)

    # Find the index of the closest standard for each pixel
    closest_indices = np.argmin(distances, axis=1)

    # Create the classified image and apply the mask
    classified_image = np.zeros((h, w), dtype=np.uint8)
    classified_image.reshape(-1)[mask] = closest_indices

    return classified_image


def apply_lut(image, category_values, lut_values, mask):
    """Applies a custom LUT to the classified image and ensures the background is black."""
    
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(lut_values):
        lut[category_values[i]] = [b, g, r]

    colored_image = cv2.LUT(cv2.merge([image] * 3), lut)
    #Ensure the background is black
    colored_image[mask == 0] = [0, 0, 0]

    return colored_image

def colour_grading(image, muscle_mask, marbling_mask, output_dir, image_id, reference_path):
    """Performs color grading on the lean muscle area (excluding marbling) and saves results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gets the lean mask (muscle area excluding marbling)
    lean_mask = cv2.subtract(muscle_mask, marbling_mask)
    standard_img = execute_color_standardization(image, reference_path)
    
    # Performs vectorized color analysis for Canadian standards
    canadian_classified_standard = classify_rgb_vectorized(standard_img, canadian_rgb_standard, lean_mask)
    # Applies LUT for visualization with a black background  
    canadian_lut_image_standard = apply_lut(canadian_classified_standard, list(range(7)), canadian_rgb_standard, lean_mask)  


    # Save results
    base_output_dir = os.path.join(output_dir, image_id)
    os.makedirs(base_output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_canadian_lut_STANDARDIZED.png"), canadian_lut_image_standard)
    #cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_STANDARDIZED.png"), standard_img) These images tend to use a lot of storage so keep them commented unless testing.

    return canadian_classified_standard, lean_mask

def save_colouring_csv(id_list, canadian_classified_list, lean_mask_list, output_csv_path):
    """Save the color analysis results for multiple images to a CSV file, ensuring all standards are represented."""
    standards = [f"CdnStd{i}" for i in range(7)]
    all_data = []
    
    for image_id, classified, mask in zip(id_list, canadian_classified_list, lean_mask_list):
        if mask is None:
            continue
        total_pixels = np.count_nonzero(mask)
        counts = dict(zip(*np.unique(classified[mask > 0], return_counts=True))) if total_pixels else {}
        for i, standard in enumerate(standards):
            count = counts.get(i, 0)
            all_data.append({
                "image_id": image_id,
                "standard": standard,
                "pixel_count": count,
                "total_pixel_count": total_pixels,
                "percentage": round((count / total_pixels) * 100, 2) if total_pixels else 0.00
            })
    
    pd.DataFrame(all_data).to_csv(output_csv_path, index=False)