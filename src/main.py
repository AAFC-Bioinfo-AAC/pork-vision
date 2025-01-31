import argparse
import numpy as np
from ultralytics import YOLO
from utils.preprocess import mask_selector, append_None_values_to_measurement_lists, convert_contours_to_image
from utils.orientation import orient_muscle_and_fat_using_adjacency
from utils.measurement import measure_longest_horizontal_segment, find_midline_using_fat_extremes, measure_vertical_segment, extend_vertical_line_to_fat
from utils.postprocess import save_annotated_image, save_results_to_csv, print_table_of_measurements, extract_image_id


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
##########################################################
# Step 1: Preprocessing
##########################################################
    args = parse_args()

    model = YOLO(args.model_path)
    results = model(args.image_path, save=True, project=args.segment_path)

    # Lists to store measurements
    id_list = []
    muscle_width_list = []
    muscle_depth_list = []
    fat_depth_list = []

    for image_result in results:
        print("\nPre-processing:", image_result.path)

        # Extract bounding boxes and segmentation masks for muscle & fat
        muscle_bbox, muscle_mask, fat_bbox, fat_mask = mask_selector(image_result)

        if muscle_bbox is None or fat_bbox is None:
            append_None_values_to_measurement_lists(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, image_result)
            continue # Next image

        # Convert muscle & fat contours into binary masks
        muscle_binary_mask = convert_contours_to_image(muscle_mask, image_result.orig_shape)
        fat_binary_mask = convert_contours_to_image(fat_mask, image_result.orig_shape)

##########################################################
# Step 2: Orientation
##########################################################
        rotated_image, rotated_muscle_mask, rotated_fat_mask, final_angle = orient_muscle_and_fat_using_adjacency(
            original_image=image_result.orig_img,
            muscle_mask=muscle_binary_mask,
            fat_mask=fat_binary_mask
        )

##########################################################
# Step 3: Measurement
##########################################################
        # Measure muscle width
        leftmost, rightmost = measure_longest_horizontal_segment(rotated_muscle_mask)
        if leftmost is None or rightmost is None:
            print("Failed to find muscle width, skipping measurement.")
            append_None_values_to_measurement_lists(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, image_result)
            continue  # Skip this image
        muscle_width = np.linalg.norm(np.array(leftmost) - np.array(rightmost))

        # Determine midline using fat mask
        midline_position, midline_point = find_midline_using_fat_extremes(rotated_fat_mask)
        if midline_position is None:
            print("Failed to determine midline, skipping measurement.")
            append_None_values_to_measurement_lists(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, image_result)
            continue  # Skip this image

        # Measure muscle depth using 7cm offset
        muscle_depth_start, muscle_depth_end = measure_vertical_segment(rotated_muscle_mask, midline_position)
        if muscle_depth_start is None or muscle_depth_end is None:
            print("Failed to measure muscle depth, skipping measurement.")
            append_None_values_to_measurement_lists(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, image_result)
            continue  # Skip this image
        muscle_depth = np.linalg.norm(np.array(muscle_depth_start) - np.array(muscle_depth_end))

        # Measure fat depth at the same x-coordinate as muscle depth
        fat_depth_start, fat_depth_end = extend_vertical_line_to_fat(rotated_fat_mask, muscle_depth_start[0])
        if fat_depth_start is None or fat_depth_end is None:
            print("Failed to measure fat depth, setting to zero.")
            fat_depth = 0
        else:
            fat_depth = np.linalg.norm(np.array(fat_depth_start) - np.array(fat_depth_end))

##########################################################
# Step 4: Postprocessing
##########################################################
        # Save measurement results
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
            image_result.path,
            args.output_path
        )

    # Save results in CSV and display in table
    save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, args.results_csv)
    print_table_of_measurements(args.results_csv)


if __name__ == "__main__":
    main()