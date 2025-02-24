#!/usr/bin/env python
from utils.imports import *
import argparse
import concurrent.futures
from ultralytics import YOLO
from utils.preprocess import (
    mask_selector,
    append_None_values_to_measurement_lists,
    convert_contours_to_image,
)
from utils.orientation import orient_muscle_and_fat_using_adjacency
from utils.marbling import process_marbling,save_marbling_csv
from utils.colouring import colour_grading, save_colouring_csv
from utils.measurement import (
    measure_longest_horizontal_segment,
    find_midline_using_fat_extremes,
    measure_vertical_segment,
    extend_vertical_line_to_fat,
    get_muscle_rotation_angle,
)
from utils.postprocess import (
    save_annotated_image,
    save_results_to_csv,
    print_table_of_measurements,
    extract_image_id,
    save_to_roi
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PorkVision Inference and Analysis")
    parser.add_argument("--model_path", type=str, default="src/models/last.pt")
    parser.add_argument("--image_path", type=str, default="data/raw_images/")
    parser.add_argument("--segment_path", type=str, default="output/segment")
    parser.add_argument("--output_path", type=str, default="output/annotated_images")
    parser.add_argument("--results_csv", type=str, default="output/results.csv")
    parser.add_argument("--rois_path", type=str, default="output/rois")
    parser.add_argument("--marbling_csv", type=str, default="output/marbling_percentage.csv")
    parser.add_argument("--colouring_path", type=str, default="output/colouring")
    parser.add_argument("--colouring_csv", type=str, default="output/colour_summary.csv")
    parser.add_argument("--standard_color_csv", type=str, default="output/colour_standardized_summary.csv")
    parser.add_argument("--reference_path", type=str, default="data/reference_images/2704_LdLeanColor.JPG")
    parser.add_argument("--marbling_path", type=str, default="output/marbling")
    return parser.parse_args()


def process_image(image_path, args):
    """Process an image: Run YOLO inference, extract measurements, and save annotated output."""
    try:
        print("\nProcessing:", image_path)

        # Each process gets its own YOLO model instance (avoids multi-threading issues)
        model = YOLO(args.model_path)

        # Step 1: YOLO Inference
        results = model(image_path, save=True, project=args.segment_path)[0]

        # Step 2: Preprocessing
        muscle_bbox, muscle_mask, fat_bbox, fat_mask = mask_selector(results)
        if muscle_bbox is None or fat_bbox is None:
            return extract_image_id(image_path), None, None, None, None, None, None, None, None, None

        muscle_binary_mask = convert_contours_to_image(muscle_mask, results.orig_shape)
        fat_binary_mask = convert_contours_to_image(fat_mask, results.orig_shape)
        #cv2.imwrite("muscle_binary", muscle_mask)

        # Step 3: Orientation
        rotated_image, rotated_muscle_mask, rotated_fat_mask, final_angle = orient_muscle_and_fat_using_adjacency(
            results.orig_img, muscle_binary_mask, fat_binary_mask
        )

        # Step 4: Marbling Extraction
        image_id = extract_image_id(image_path)
        marbling_mask, marbling_percentage = process_marbling(rotated_image, rotated_muscle_mask, args.marbling_path, base_filename=image_id)

        # Step 5: Perform color grading
        # NOTE results.orig_image is used in favor against rotated image to solve issues with Standardization.
        canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask = colour_grading(rotated_image, rotated_muscle_mask, marbling_mask, args.colouring_path, image_id, args.reference_path)

        # Step 6: Measurement
        muscle_width_start, muscle_width_end = measure_longest_horizontal_segment(rotated_muscle_mask)
        if muscle_width_start is None or muscle_width_end is None:
            return extract_image_id(image_path), None, None, None, marbling_percentage, canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask
        muscle_width = np.linalg.norm(np.array(muscle_width_start) - np.array(muscle_width_end))

        midline_position, midline_point = find_midline_using_fat_extremes(rotated_fat_mask)
        if midline_position is None:
            return extract_image_id(image_path), muscle_width, None, None, marbling_percentage, canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask

        angle = get_muscle_rotation_angle(rotated_muscle_mask)
        if angle is None:
            return extract_image_id(image_path), muscle_width, None, None, marbling_percentage, canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask

        muscle_depth_start, muscle_depth_end = measure_vertical_segment(rotated_muscle_mask, midline_position, angle)
        if muscle_depth_start is None or muscle_depth_end is None:
            return extract_image_id(image_path), muscle_width, None, None, marbling_percentage, canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask
        muscle_depth = np.linalg.norm(np.array(muscle_depth_start) - np.array(muscle_depth_end))

        fat_depth_start, fat_depth_end = extend_vertical_line_to_fat(rotated_fat_mask, (muscle_depth_start, muscle_depth_end))
        if fat_depth_start is None or fat_depth_end is None:
            return extract_image_id(image_path), muscle_width, muscle_depth, None, marbling_percentage, canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask
        fat_depth = np.linalg.norm(np.array(fat_depth_start) - np.array(fat_depth_end))

        # Step 7: Save annotated image
        save_annotated_image(
            rotated_image, (muscle_width_start, muscle_width_end), (muscle_depth_start, muscle_depth_end),
            (fat_depth_start, fat_depth_end), image_path, args.output_path
        )

        save_to_roi(
            muscle_width_start,
            muscle_width_end,
            muscle_depth_start,
            muscle_depth_end,
            fat_depth_start,
            fat_depth_end,
            image_id=extract_image_id(image_path),
            rois_folder=args.rois_path
        )

        return image_id, muscle_width, muscle_depth, fat_depth, marbling_percentage, canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return extract_image_id(image_path), None, None, None, None, None, None, None, None, None


def main():
    args = parse_args()

    # Step 1: Get all image paths
    image_paths = sorted([os.path.join(args.image_path, img) for img in os.listdir(args.image_path)])

    # Step 2: Parallel Processing
    id_list, muscle_width_list, muscle_depth_list, fat_depth_list, marbling_percentage_list = [], [], [], [], []
    canadian_classified_list, japanese_classified_list, canadian_classified_standard_list, japanese_classified_standard_list, lean_mask_list = [], [], [], [], []
    max_workers = min(4, os.cpu_count() // 2)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, img_path, args): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            img_id, muscle_width, muscle_depth, fat_depth, marbling_percentage, canadian_classified, japanese_classified, canadian_classified_standard, japanese_classified_standard, lean_mask = result
            id_list.append(img_id)
            muscle_width_list.append(muscle_width)
            muscle_depth_list.append(muscle_depth)
            fat_depth_list.append(fat_depth)
            marbling_percentage_list.append(marbling_percentage)
            canadian_classified_list.append(canadian_classified)
            japanese_classified_list.append(japanese_classified)
            canadian_classified_standard_list.append(canadian_classified_standard)
            japanese_classified_standard_list.append(japanese_classified_standard)
            lean_mask_list.append(lean_mask)

            
    # Step 3: Save and display results
    save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, args.results_csv)
    save_marbling_csv(id_list, marbling_percentage_list, args.marbling_csv)
    print_table_of_measurements(args.results_csv)
    print_table_of_measurements(args.marbling_csv)
    
    # OPTIONAL: Comment these out to improve performance
    save_colouring_csv(id_list, canadian_classified_list, japanese_classified_list, lean_mask_list, args.colouring_csv)
    save_colouring_csv(id_list, canadian_classified_standard_list, japanese_classified_standard_list, lean_mask_list, args.standard_color_csv)
    print_table_of_measurements(args.colouring_csv)
    print_table_of_measurements(args.standard_color_csv)

if __name__ == "__main__":
    main()