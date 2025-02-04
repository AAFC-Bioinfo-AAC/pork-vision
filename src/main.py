import argparse
import concurrent.futures
import os
import numpy as np
from ultralytics import YOLO
from utils.preprocess import (
    mask_selector,
    append_None_values_to_measurement_lists,
    convert_contours_to_image,
)
from utils.orientation import orient_muscle_and_fat_using_adjacency
from utils.measurement import (
    measure_longest_horizontal_segment,
    find_midline_using_fat_extremes,
    measure_vertical_segment,
    extend_vertical_line_to_fat,
)
from utils.postprocess import (
    save_annotated_image,
    save_results_to_csv,
    print_table_of_measurements,
    extract_image_id,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run PorkVision Inference and Analysis")
    parser.add_argument("--model_path", type=str, default="src/models/last.pt")
    parser.add_argument("--image_path", type=str, default="data/raw_images/")
    parser.add_argument("--segment_path", type=str, default="output/segment")
    parser.add_argument("--output_path", type=str, default="output/annotated_images")
    parser.add_argument("--results_csv", type=str, default="output/results.csv")
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
            return extract_image_id(image_path), None, None, None

        muscle_binary_mask = convert_contours_to_image(muscle_mask, results.orig_shape)
        fat_binary_mask = convert_contours_to_image(fat_mask, results.orig_shape)

        # Step 3: Orientation
        rotated_image, rotated_muscle_mask, rotated_fat_mask, final_angle = orient_muscle_and_fat_using_adjacency(
            results.orig_img, muscle_binary_mask, fat_binary_mask
        )

        # Step 4: Measurement
        leftmost, rightmost = measure_longest_horizontal_segment(rotated_muscle_mask)
        if leftmost is None or rightmost is None:
            return extract_image_id(image_path), None, None, None
        muscle_width = np.linalg.norm(np.array(leftmost) - np.array(rightmost))

        midline_position, midline_point = find_midline_using_fat_extremes(rotated_fat_mask)
        if midline_position is None:
            return extract_image_id(image_path), None, None, None

        muscle_depth_start, muscle_depth_end = measure_vertical_segment(rotated_muscle_mask, midline_position)
        if muscle_depth_start is None or muscle_depth_end is None:
            return extract_image_id(image_path), None, None, None
        muscle_depth = np.linalg.norm(np.array(muscle_depth_start) - np.array(muscle_depth_end))

        fat_depth_start, fat_depth_end = extend_vertical_line_to_fat(rotated_fat_mask, (muscle_depth_start, muscle_depth_end))
        if fat_depth_start is None or fat_depth_end is None:
            return extract_image_id(image_path), None, None, None
        fat_depth = np.linalg.norm(np.array(fat_depth_start) - np.array(fat_depth_end))

        # Step 5: Save annotated image
        save_annotated_image(
            rotated_image, (leftmost, rightmost), (muscle_depth_start, muscle_depth_end),
            (fat_depth_start, fat_depth_end), image_path, args.output_path
        )

        return extract_image_id(image_path), muscle_width, muscle_depth, fat_depth

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return extract_image_id(image_path), None, None, None


def main():
    args = parse_args()

    # Step 1: Get all image paths
    image_paths = sorted([os.path.join(args.image_path, img) for img in os.listdir(args.image_path)])

    # Step 2: Parallel Processing
    id_list, muscle_width_list, muscle_depth_list, fat_depth_list = [], [], [], []
    max_workers = min(4, os.cpu_count() // 2)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, img_path, args): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(futures):
            img_id, muscle_width, muscle_depth, fat_depth = future.result()
            id_list.append(img_id)
            muscle_width_list.append(muscle_width)
            muscle_depth_list.append(muscle_depth)
            fat_depth_list.append(fat_depth)

    # Step 3: Save and display results
    save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, args.results_csv)
    print_table_of_measurements(args.results_csv)


if __name__ == "__main__":
    main()