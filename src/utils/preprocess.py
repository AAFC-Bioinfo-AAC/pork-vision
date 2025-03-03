from skimage.draw import polygon2mask
from ultralytics.data.utils import polygon2mask

def mask_selector(current_image, confidence_threshold=0.4):
    """
    Selects the best masks for muscle and fat classes based on confidence scores.

    Parameters:
    - current_image (object): YOLOv8 inference result object.
    - confidence_threshold (float): Minimum confidence score for a valid detection.

    Returns:
    - tuple: (muscle_bbox, muscle_mask, fat_bbox, fat_mask), where each is either a valid numpy array or None.
    """

    # Extract class labels and confidence scores
    classes = current_image.boxes.cls.numpy().astype("int")
    confidences = current_image.boxes.conf.numpy()

    # Ensure masks exist before accessing them
    has_masks = hasattr(current_image, "masks") and current_image.masks is not None

    # Initialize placeholders for best muscle & fat detections
    muscle_bbox, muscle_mask, fat_bbox, fat_mask = None, None, None, None
    confidence_muscle, confidence_fat = 0, 0

    # Loop through detections and pick the highest confidence muscle and fat masks
    for j, cls in enumerate(classes):
        if confidences[j] < confidence_threshold:
            continue  # Skip detections below the threshold

        if cls == 0 and confidences[j] > confidence_muscle:  # Muscle class
            confidence_muscle = confidences[j]
            muscle_bbox = current_image.boxes[j].xyxy  # Bounding box
            muscle_mask = current_image.masks[j].xy if has_masks else None  # Mask contour

        elif cls == 1 and confidences[j] > confidence_fat:  # Fat class
            confidence_fat = confidences[j]
            fat_bbox = current_image.boxes[j].xyxy  # Bounding box
            fat_mask = current_image.masks[j].xy if has_masks else None  # Mask contour

    # If any of the masks are missing, return None
    if muscle_bbox is None or muscle_mask is None or fat_bbox is None or fat_mask is None:
        print(f"Skipping image - Missing valid muscle or fat mask.")
        return None, None, None, None

    return muscle_bbox[0], muscle_mask, fat_bbox[0], fat_mask

def append_None_values_to_measurement_lists(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, image_result):
    """
    Appends None values to measurement lists when an image is skipped.

    Parameters:
    - id_list (list): List of image IDs.
    - muscle_width_list (list): List of muscle width values.
    - muscle_depth_list (list): List of muscle depth values.
    - fat_depth_list (list): List of fat depth values.
    - image_result (object): YOLO inference result object.
    """
    image_id = image_result.path.split("/")[-1].split(".")[0]
    id_list.append(image_id)
    muscle_width_list.append(None)
    muscle_depth_list.append(None)
    fat_depth_list.append(None)

def convert_contours_to_image(contours, orig_shape):
    """
    Converts YOLO inferred contours into a binary mask.

    Parameters:
    contours (list): List of contour points.
    orig_shape (tuple): Original shape of the image.

    Returns:
    numpy.ndarray: Binary mask with the contours drawn.
    """
    mask = polygon2mask(
        orig_shape,
        [contours],
        color=255,
        downsample_ratio=1,
    )
    return mask