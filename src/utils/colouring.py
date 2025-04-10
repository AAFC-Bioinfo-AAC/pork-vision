from utils.imports import *
from scipy import stats
import numpy as np

#canadian_rgb_standard = np.array([
#    (171, 90, 98), # C0
#    (183, 103, 109), # C1
#    (192, 124, 123), # C2
#    (198, 142, 136), # C3
#    (208, 161, 151), # C4
#    (212, 174, 160), # C5
#    (219, 188, 169),   # C6
#], dtype=np.float32)



# RGB values for Canadian lean color standards made from 102,103,104,105,107,109,110 2024 images
class_to_std = {0: "Canadian_Std6",
                1 : "Canadian_Std5",
                2 : "Canadian_Std4",
                3 : "Canadian_Std3",
                4 : "Canadian_Std2", 
                5 : "Canadian_Std1", 
                6 : "Canadian_Std0",}

def get_mode_rgb(image, bbox):
    try:
        x_min, y_min, x_max, y_max = bbox.xyxy[0]  # Focus interest on the bounding box.
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        focus = image[y_min:y_max, x_min:x_max].reshape(-1,3)  #Crop into the bounding box and reshape the image into 2D array consisting of rows of pixel containing RGB
        mode_rgb = stats.mode(focus, axis=0).mode
        return mode_rgb
    except:
        return (0,0,0)
    
def insertion_sort(canadian_standard_unsorted):
    for i in range(1, len(canadian_standard_unsorted)):
        key = canadian_standard_unsorted[i]
        #print(f"Current key is {key}")
        index = i - 1
        #print(f"Key - 1 (index) is {index}")

        while index >= 0 and key[0]<canadian_standard_unsorted[index][0]:
            #print(f"Swapping since index is {index} and key ({key}) is less than Canadian_Unsorted[index][0]: {canadian_standard_unsorted[index][0]}")
            canadian_standard_unsorted[index + 1] = canadian_standard_unsorted[index]
            index = index - 1
        #print(f"Setting unsorted canadian_standard index+1[0] ({canadian_standard_unsorted[index+1][0]} to key ({key}))")
        canadian_standard_unsorted[index+1] = key
    #print()
    #print(f"Sorted array in Ascending order {canadian_standard_unsorted}")
    for sublist in canadian_standard_unsorted:
        del sublist[0]
    #print(canadian_standard_unsorted)
    return canadian_standard_unsorted

############################
#####ANALYSIS FUNCTIONS#####
############################
def classify_rgb_vectorized(image, standards, lean_mask):
    """Vectorized classification of RGB pixels using Euclidean distance."""
    # Convert the image to RGB (if it's in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    standard_0 = standards[-1]
    print(f"Canadian Standard 0 is: {standard_0}")
    cutoff = standard_0 + 0.2
    print(f"The cutoff is {cutoff}")
    array = np.vstack([standards, cutoff])
    print(f"The new array is {array}")
    
    # Reshape the image to a (h*w, 3) matrix of RGB values
    h, w, _ = image.shape
    image_rgb_reshaped = image_rgb.reshape(-1, 3)

    # Create a mask to filter out pixels where lean_mask == 0
    mask = lean_mask.reshape(-1) > 0

    # Compute the distances between each pixel and each standard
    distances = np.linalg.norm(image_rgb_reshaped[mask][:, np.newaxis] - array, axis=2)

    # Find the index of the closest standard for each pixel
    closest_indices = np.argmin(distances, axis=1)

    # Create the classified image and apply the mask
    classified_image = np.zeros((h, w), dtype=np.uint8)
    classified_image.reshape(-1)[mask] = closest_indices

    return classified_image,array


def apply_lut(image, category_values, lut_values, mask):
    """Applies a custom LUT to the classified image and ensures the background is black."""
    
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    lut_values = enumerate(lut_values)
    print(lut_values)
    for i, (r, g, b) in lut_values:
        print(f"INDEX {i} : RGB {r, g, b}")
        if i == 7:
            lut[category_values[i]] = [0,0,0]
            continue
        lut[category_values[i]] = [b, g, r]

    colored_image = cv2.LUT(cv2.merge([image] * 3), lut)
    #Ensure the background is black
    colored_image[mask == 0] = [0, 0, 0]

    return colored_image

def colour_grading(image, muscle_mask, marbling_mask, output_dir, image_id, model):
    """Performs color grading on the lean muscle area (excluding marbling) and saves results."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Gets the lean mask (muscle area excluding marbling)
    lean_mask = cv2.subtract(muscle_mask, marbling_mask)


    canadian_standard_unsorted = []
    result = model.predict(image, save=False)[0]
    detection_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #print(f"{image_id}_LdLeanColor.JPG")
    if result == None:
        return None, lean_mask, 'Y'
    for box in result.boxes:
        duplicate = False
        class_id = int(box.cls[0])
        for items in canadian_standard_unsorted:
            if items[0] == class_id:
                duplicate = True
        if duplicate == True:
            continue
        #confidence = box.conf[0]
        #print(f"Class ID: {class_id}, Confidence: {confidence}")
        mode_rgb = get_mode_rgb(detection_image, box)
        id_rgb = [class_id, mode_rgb]
        #print(f"The mode rgb for {class_to_std[class_id]} is {mode_rgb}")
        #print(id_rgb)
        canadian_standard_unsorted.append(id_rgb)
    if len(canadian_standard_unsorted) != 7:
       return None, lean_mask, 'Y'
    #print(canadian_standard_unsorted)
    canadian_standard_sorted = insertion_sort(canadian_standard_unsorted)
    canadian_array = np.array([item[0] for item in canadian_standard_sorted], dtype=np.float32)
    print(f"{image_id}_LdLeanColor.JPG = {canadian_array}")
    
    # Performs vectorized color analysis for Canadian standards
    canadian_classified,canadian_array = classify_rgb_vectorized(image, canadian_array, lean_mask)
    # Applies LUT for visualization with a black background  
    canadian_lut_image = apply_lut(canadian_classified, list(range(8)), canadian_array, lean_mask)  

    # Save results
    base_output_dir = os.path.join(output_dir, image_id)
    os.makedirs(base_output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(base_output_dir, f"{image_id}_canadian_lut.png"), canadian_lut_image)
    save_path = f'{base_output_dir}/{image_id}_Color_Detect.jpg'
    result.save(save_path)

    return canadian_classified, lean_mask, None

def save_colouring_csv(id_list, canadian_classified_list, lean_mask_list, output_csv_path, colour_outlier_list):
    """Save the color analysis results for multiple images to a CSV file, ensuring all standards are represented."""
    standards = [f"CdnStd{i}" for i in range(7)]
    all_data = []
    
    for image_id, classified, mask, outlier in zip(id_list, canadian_classified_list, lean_mask_list, colour_outlier_list):
        if mask is None or outlier is not None:
            all_data.append({
                    "image_id": image_id,
                    "standard": '?',
                    "pixel_count": '?',
                    "total_pixel_count": '?',
                    "percentage": '?',
                    "outlier?": outlier
                })
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