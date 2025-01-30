import argparse
import os
import re
import math
import cv2
import ellipse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from csv import reader
from skimage import draw, measure
from skimage.draw import polygon2mask
from ultralytics.data.utils import polygon2mask
from ultralytics import YOLO


def mask_selector(current_image, confidence_threshold=0.5):
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
    Applies morphological closing to smooth small holes/gaps in the binary mask.

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
    return cv2.warpAffine(image, rot_matrix, (w, h))

def isolate_adjacent_fat(muscle_mask, fat_mask, dilation_size=15, min_area=500):
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
    # 1. Dilate the muscle mask
    dilated_muscle = dilate_mask(muscle_mask, kernel_size=dilation_size)

    # 2. Intersection with the fat mask
    adjacent_fat = cv2.bitwise_and(dilated_muscle, fat_mask)

    # 3. Smooth the adjacent fat region
    adjacent_fat = preprocess_mask(adjacent_fat)

    # 4. Find largest valid contour
    return find_largest_contour(adjacent_fat, min_area=min_area)

def orient_muscle_and_fat_using_adjacency(original_image, muscle_mask, fat_mask):
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
        print("No muscle found. Skipping orientation.")
        return original_image, muscle_mask, fat_mask, 0
    muscle_contour = max(muscle_contours, key=cv2.contourArea)
    mx, my, mw, mh = cv2.boundingRect(muscle_contour)
    muscle_center_x = mx + mw / 2
    muscle_center_y = my + mh / 2

    # 2. Isolate adjacent portion of the fat
    adjacent_fat_box = isolate_adjacent_fat(muscle_mask, fat_mask, dilation_size=15, min_area=500)

    # 3. Check if fat is already on top (when adjacent_fat_box is missing)
    if adjacent_fat_box is None:
        # Compare muscle bounding box position with fat bounding box
        fat_contours, _ = cv2.findContours(fat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if fat_contours:
            fat_contour = max(fat_contours, key=cv2.contourArea)
            fx, fy, fw, fh = cv2.boundingRect(fat_contour)
            fat_center_y = fy + fh / 2

            if fat_center_y < muscle_center_y:
                print("Fat already correctly positioned on top. No rotation needed.")
                return original_image, muscle_mask, fat_mask, 0
        
        print("No valid adjacent fat region detected. Skipping orientation.")
        return original_image, muscle_mask, fat_mask, 0

    # 4. Normal orientation logic (only if `adjacent_fat_box` was found)
    fx, fy, fw, fh = adjacent_fat_box
    fat_center_x = fx + fw / 2
    fat_center_y = fy + fh / 2

    dx = fat_center_x - muscle_center_x
    dy = fat_center_y - muscle_center_y

    final_angle = 0
    if dy > 0 and abs(dy) >= abs(dx):
        print("Fat (adjacent region) below muscle. Rotating 180°.")
        final_angle = 180
    elif dx > 0 and abs(dx) > abs(dy):
        print("Fat (adjacent region) on the right. Rotating +90°.")
        final_angle = 90
    elif dx < 0 and abs(dx) > abs(dy):
        print("Fat (adjacent region) on the left. Rotating -90°.")
        final_angle = -90
    else:
        print("Fat (adjacent region) already on top. No rotation needed.")

    # 5. Apply final rotation
    rotated_image = rotate_image(original_image, final_angle)
    rotated_muscle_mask = rotate_image(muscle_mask, final_angle)
    rotated_fat_mask = rotate_image(fat_mask, final_angle)

    return rotated_image, rotated_muscle_mask, rotated_fat_mask, final_angle





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
    Determines the midline (bone side) using the vertical position of the fat mask’s leftmost and rightmost points.

    The **higher** point (lower Y value) is the **bone side**.

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

    # Find leftmost and rightmost fat points
    leftmost = tuple(fat_contour[fat_contour[:, :, 0].argmin()][0])  # (x, y)
    rightmost = tuple(fat_contour[fat_contour[:, :, 0].argmax()][0])  # (x, y)

    # The higher (smaller Y) point is the midline (bone side)
    if leftmost[1] < rightmost[1]:  # Leftmost is higher up
        return "LEFT", leftmost
    else:
        return "RIGHT", rightmost

def measure_longest_vertical_segment(muscle_mask):
    """
    Finds the longest vertical segment in the muscle mask and returns its start & end points.

    Returns:
    - tuple: ((x, y1), (x, y2)) representing the longest vertical segment.
    """
    contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None  # No contour found

    muscle_contour = max(contours, key=cv2.contourArea)

    # Find the highest (min y) and lowest (max y) points for each x value
    unique_xs = np.unique(muscle_contour[:, :, 0])

    longest_segment = None
    max_length = 0

    for x in unique_xs:
        y_values = muscle_contour[muscle_contour[:, :, 0] == x][:, 1]
        top_y = np.min(y_values)
        bottom_y = np.max(y_values)
        length = bottom_y - top_y

        if length > max_length:
            max_length = length
            longest_segment = ((x, top_y), (x, bottom_y))

    return longest_segment

def extend_vertical_line_to_fat(fat_mask, muscle_depth_x):
    """
    Measures the fat depth at the same x-coordinate used for muscle depth.

    Parameters:
    - fat_mask (numpy.ndarray): Binary mask of the fat (1 where fat exists, 0 elsewhere).
    - muscle_depth_x (int): The x-coordinate used for muscle depth measurement.

    Returns:
    - tuple: ((x, y1), (x, y2)) representing the fat depth line.
    """
    if muscle_depth_x is None or muscle_depth_x < 0 or muscle_depth_x >= fat_mask.shape[1]:
        print("extend_vertical_line_to_fat: Invalid x-coordinate. Skipping measurement.")
        return None, None

    # Extract the fat mask column at muscle_depth_x
    column_pixels = fat_mask[:, muscle_depth_x]

    # Find nonzero (fat) pixel locations
    fat_pixels = np.where(column_pixels > 0)[0]  # Get y-values where fat exists

    if fat_pixels.size == 0:
        print(f"extend_vertical_line_to_fat: No fat detected at x = {muscle_depth_x}. Skipping measurement.")
        return None, None

    # Get the min (top) and max (bottom) y-coordinates of the fat in this column
    top_fat_y = np.min(fat_pixels)
    bottom_fat_y = np.max(fat_pixels)

    return (muscle_depth_x, bottom_fat_y), (muscle_depth_x, top_fat_y)





def extract_image_id(image_path):
    """
    Extracts the image ID from a filename.

    Parameters:
    - image_path (str): Full path of the image file.

    Returns:
    - str: Extracted image ID (e.g., "1701").
    """

    # Get the filename without path
    filename = os.path.basename(image_path)

    # Remove the extension (e.g., .JPG, .PNG)
    filename_no_ext = os.path.splitext(filename)[0]

    return filename_no_ext

def save_annotated_image(image, muscle_width, muscle_depth, fat_depth, midline_position, image_path, output_path):
    """
    Draws measurement lines on the image and saves it.

    Parameters:
    - image (numpy.ndarray): The rotated image.
    - muscle_width (tuple): (leftmost, rightmost) points defining the muscle width.
    - muscle_depth (tuple): (start, end) points defining the muscle depth.
    - fat_depth (tuple): (start, end) points defining the fat depth.
    - midline_position (str): "LEFT" or "RIGHT" indicating which side is the midline.
    - image_path (str): Original image file path (to extract filename).
    - output_path (str): Directory to save annotated images.
    """

    # Create a copy of the image for annotation
    annotated_image = image.copy()

    # Define colors (BGR format)
    width_color = (0, 255, 0)  # Green for muscle width
    depth_color = (0, 0, 255)  # Red for muscle depth
    fat_color = (255, 0, 0)  # Blue for fat depth
    thickness = 5

    # Draw muscle width line
    if muscle_width:
        cv2.line(annotated_image, muscle_width[0], muscle_width[1], width_color, thickness)

    # Draw muscle depth line
    if muscle_depth:
        cv2.line(annotated_image, muscle_depth[0], muscle_depth[1], depth_color, thickness)
    else:
        print("no muscle depth")

    # Draw fat depth line
    if fat_depth:
        cv2.line(annotated_image, fat_depth[0], fat_depth[1], fat_color, thickness)
    else:
        print("no fat depth")

    # Extract filename and define output path
    filename = os.path.basename(image_path)
    output_file = os.path.join(output_path, f"annotated_{filename}")

    # Save the annotated image
    cv2.imwrite(output_file, annotated_image)

    print(f"Annotated image saved: {output_file}" + "\n")

def save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, output_csv_path):
    """
    Saves the measurement results to a CSV file.

    Parameters:
    - id_list (list): List of image IDs.
    - muscle_width_list (list): List of measured muscle widths.
    - muscle_depth_list (list): List of measured muscle depths.
    - fat_depth_list (list): List of measured fat depths.
    - output_csv_path (str): Path to save the CSV file.
    """

    df = pd.DataFrame({
        "image_id": id_list,
        "muscle_width_px": muscle_width_list,
        "muscle_depth_px": muscle_depth_list,
        "fat_depth_px": fat_depth_list
    })

    # Convert measurements to millimeters (assuming 140 pixels = 1 cm)
    conversion_factor = 10 / 140  # 10 mm per cm, 140 px per cm
    df_mm = df.iloc[:, 1:] * conversion_factor
    df_mm.columns = ["muscle_width_mm", "muscle_depth_mm", "fat_depth_mm"]

    # Concatenate pixel and mm measurements
    df = pd.concat([df, df_mm], axis=1)

    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False)

    print(f"Results saved to: {output_csv_path}")

def print_table_of_measurements(results_csv_path):
    """
    Reads the CSV file and prints the results in a formatted table.

    Parameters:
    - results_csv_path (str): Path to the CSV file containing measurement results.
    """

    # Load the CSV file
    try:
        df = pd.read_csv(results_csv_path)
        
        # Print the table using tabulate
        print("\nMeasurement Results:")
        print(tabulate(df, headers="keys", tablefmt="pipe", showindex=False))

    except Exception as e:
        print(f"Error reading results CSV: {e}")



    

def parse_args():
    parser = argparse.ArgumentParser(description="Run PorkVision Inference and Analysis")
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/models/last.pt",
        help="Path to the YOLO model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/raw_images/",
        help="Path to the raw images",
    )
    parser.add_argument(
        "--segment_path",
        type=str,
        default="output/segment",
        help="Path to save the segmented images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/annotated_images",
        help="Path to save the output",
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        default="output/results.csv",
        help="Path to save the results CSV",
    )
    
    return parser.parse_args()


def main():
    args = parse_args()

    model = YOLO(args.model_path)
    results = model(args.image_path, save=True, project=args.segment_path)

    # Lists to store measurements
    id_list = []
    muscle_width_list = []
    muscle_depth_list = []
    fat_depth_list = []

    for image_result in results:
        print("\nProcessing:", image_result.path)

        # Extract bounding boxes and segmentation masks for muscle & fat
        muscle_bbox, muscle_mask, fat_bbox, fat_mask = mask_selector(image_result)

        if muscle_bbox is None or fat_bbox is None:
            append_None_values_to_measurement_lists(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, image_result)
            continue # Next image

        # Convert muscle & fat contours into binary masks
        muscle_binary_mask = convert_contours_to_image(muscle_mask, image_result.orig_shape)
        fat_binary_mask = convert_contours_to_image(fat_mask, image_result.orig_shape)

        # Orient fat on top
        rotated_image, rotated_muscle_mask, rotated_fat_mask, final_angle = orient_muscle_and_fat_using_adjacency(
            original_image=image_result.orig_img,
            muscle_mask=muscle_binary_mask,
            fat_mask=fat_binary_mask
        )

        # Measure muscle width
        leftmost, rightmost = measure_longest_horizontal_segment(rotated_muscle_mask)
        muscle_width = np.linalg.norm(np.array(leftmost) - np.array(rightmost))

        # Measure longest vertical muscle depth
        muscle_depth_start, muscle_depth_end = measure_longest_vertical_segment(rotated_muscle_mask)

        if muscle_depth_start and muscle_depth_end:
            muscle_depth = np.linalg.norm(np.array(muscle_depth_start) - np.array(muscle_depth_end))

            # Measure fat depth at the same x-coordinate
            fat_depth_start, fat_depth_end = extend_vertical_line_to_fat(rotated_fat_mask, muscle_depth_start[0])
            fat_depth = np.linalg.norm(np.array(fat_depth_start) - np.array(fat_depth_end)) if fat_depth_start and fat_depth_end else 0
        else:
            muscle_depth = 0
            fat_depth = 0

        # Save results
        id_list.append(extract_image_id(image_result.path))
        muscle_width_list.append(muscle_width)
        muscle_depth_list.append(muscle_depth)
        fat_depth_list.append(fat_depth)

        # Save annotated image
        save_annotated_image(
            rotated_image,
            (leftmost, rightmost),
            (muscle_depth_start, muscle_depth_end),
            (fat_depth_start, fat_depth_end),
            None,  # Midline is not needed yet
            image_result.path,
            args.output_path
        )

    save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, args.results_csv)
    print_table_of_measurements(args.results_csv)

if __name__ == "__main__":
    main()