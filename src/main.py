#!/usr/bin/env python
from utils.imports import *
import concurrent.futures
from ultralytics import YOLO
from utils.preprocess import (
    mask_selector,
    append_None_values_to_measurement_lists,
    convert_contours_to_image,
)
from utils.orientation import orient_muscle_and_fat_using_adjacency
from utils.marbling import process_marbling,save_marbling_csv
from utils.colouring import colour_grading, save_colouring_csv, create_coloring_standards
from utils.measurement import (
    measure_longest_horizontal_segment,
    find_midline_using_fat_extremes,
    measure_vertical_segment,
    extend_vertical_line_to_fat,
    get_muscle_rotation_angle,
    measure_ruler
)
from utils.postprocess import (
    save_annotated_image,
    save_results_to_csv,
    print_table_of_measurements,
    extract_image_id,
    save_to_roi
)

def debug_info(debug_messages,image_id,args):
    if args.debug == True:
        os.makedirs(f"{args.output_path}/debug", exist_ok=True)
        with open(f'{args.output_path}/debug/{image_id}_DEBUGINFO.txt', 'w') as file:
            for string in debug_messages:
                file.write(string+"\n")
            file.close()

def process_image(model, image_path, args, color_model):
    """Process an image: Run YOLO inference, extract measurements, and save annotated output."""
    try:
        outlier = None
        color_outlier = None
        image_id = extract_image_id(image_path)
        debug_messages = []
        debug_messages.append(f"{image_id}:")
        print("\nProcessing:", image_id)
        
        # Step 1: YOLO Inference
        
        results = model(image_path, save=False, retina_masks=True)[0]  # This disables automatic saving into subfolders
        # Save the result manually to the 'predict' folder
        if args.minimal == False:
            os.makedirs(f'{args.output_path}/predict', exist_ok=True)
            save_path = f'{args.output_path}/predict/{extract_image_id(image_path)}.jpg'
            results.save(save_path)  # Save the annotated image to the specified path
            debug_messages.append(f"Image is saved to {args.output_path}/predict")

        # Step 2: Preprocessing
        muscle_bbox, muscle_mask, fat_bbox, fat_mask, debug_messages = mask_selector(results, debug_messages)
        if muscle_bbox is None or fat_bbox is None:
            outlier = "Y"
            debug_messages.append(f"ERROR: Did not detect a muscle/fat bounding box.")
            debug_info(debug_messages, image_id, args)
            return extract_image_id(image_path), 0, 0, 0, 0, 0, 0, 0, 0, outlier, color_outlier, image_path, 0
        debug_messages.append("Converting muscle mask to image")
        muscle_binary_mask, debug_messages = convert_contours_to_image(muscle_mask, results.orig_shape, debug_messages)
        debug_messages.append("Converting fat mask to image")
        fat_binary_mask, debug_messages = convert_contours_to_image(fat_mask, results.orig_shape, debug_messages)
        # Step 3: Orientation
        rotated_image, rotated_muscle_mask, rotated_fat_mask, final_angle, outlier = orient_muscle_and_fat_using_adjacency(
            results.orig_img, muscle_binary_mask, fat_binary_mask, outlier
        )


        # Step 4: Conversion Factor Calculation
        conversion_factor, outlier = measure_ruler(rotated_image, image_id, outlier, args.minimal)
        if conversion_factor == None:
            debug_messages.append(f"ERROR: Conversion Factor calculation, using default.")
            conversion_factor = 10/140 #mm/px
        # Step 6: Create Canadian Standard chart using A.I model.
        canadian_standards = create_coloring_standards(rotated_image, color_model, image_id, args.output_path+'/colouring', args.minimal)
        # Step 7: Find Marbling
        marbling_mask, eroded_mask, marbling_percentage, area_px = process_marbling(rotated_image, rotated_muscle_mask, args.output_path+'/marbling', canadian_standards, args.minimal, base_filename=image_id)
        area_mm = area_px/((1/(conversion_factor))**2)
        if args.minimal == False:
            cv2.imwrite(f"{args.output_path}/marbling/{image_id}/{image_id}_fat_mask.jpg", rotated_fat_mask)

        # Step 8: Perform color grading
        canadian_classified_standard, lean_mask, color_outlier = colour_grading(
            rotated_image, eroded_mask, marbling_mask, args.output_path+'/colouring', image_id, canadian_standards, args.minimal
        )

        # Step 9: Measurement
        angle = get_muscle_rotation_angle(rotated_muscle_mask)
        if angle is None:
            outlier = "Y"
            debug_messages.append(f"ERROR: ANGLE is None")
            debug_info(debug_messages, image_id, args)
            return extract_image_id(image_path), 0, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm

        muscle_width_start, muscle_width_end = measure_longest_horizontal_segment(rotated_muscle_mask, angle)
        if muscle_width_start is None or muscle_width_end is None:
            outlier = "Y"
            debug_messages.append(f"ERROR: Muscle width is None")
            debug_info(debug_messages, image_id, args)
            return extract_image_id(image_path), 0, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm
        muscle_width = np.linalg.norm(np.array(muscle_width_start) - np.array(muscle_width_end))

        midline_position, midline_point = find_midline_using_fat_extremes(rotated_fat_mask)
        if midline_position is None:
            outlier = "Y"
            debug_messages.append(f"ERROR: Midline position is None")
            debug_info(debug_messages, image_id, args)
            return extract_image_id(image_path), muscle_width, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm


        muscle_depth_start, muscle_depth_end = measure_vertical_segment(rotated_muscle_mask, midline_position, angle, (1/(conversion_factor)*10))
        if muscle_depth_start is None or muscle_depth_end is None:
            outlier = "Y"
            debug_messages.append(f"ERROR: Muscle Depth is None")
            debug_info(debug_messages, image_id, args)
            return extract_image_id(image_path), muscle_width, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm
        muscle_depth = np.linalg.norm(np.array(muscle_depth_start) - np.array(muscle_depth_end))

        fat_depth_start, fat_depth_end = extend_vertical_line_to_fat(rotated_fat_mask, (muscle_depth_start, muscle_depth_end))
        if fat_depth_start is None or fat_depth_end is None:
            outlier = "Y"
            debug_messages.append(f"ERROR: Fat depth is None")
            debug_info(debug_messages, image_id, args)
            return extract_image_id(image_path), muscle_width, muscle_depth, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm
        fat_depth = np.linalg.norm(np.array(fat_depth_start) - np.array(fat_depth_end))

        # Step 7: Save annotated image
        if args.minimal == False:
            save_annotated_image(
                rotated_image, (muscle_width_start, muscle_width_end), (muscle_depth_start, muscle_depth_end),
                (fat_depth_start, fat_depth_end), image_path, args.output_path+'/annotated_images', args.minimal
            )

            save_to_roi(
                muscle_width_start,
                muscle_width_end,
                muscle_depth_start,
                muscle_depth_end,
                fat_depth_start,
                fat_depth_end,
                image_id=extract_image_id(image_path),
                rois_folder=args.output_path+'/rois'
            )
        
        debug_info(debug_messages, image_id, args)
        return (
            image_id,
            int(round(muscle_width)),
            int(round(muscle_depth)),
            int(round(fat_depth)),
            marbling_percentage,
            canadian_classified_standard,
            lean_mask,
            conversion_factor,
            area_px,
            outlier,
            color_outlier,
            image_path,
            area_mm
        )
            

    except Exception as e:
        debug_messages.append(f"Error processing {image_id}: {e}")
        debug_info(debug_messages, image_id, args)
        return image_id, 0, 0, 0, 0, 0, 0, 0, 0, "Y", "Y", image_path, 0


def main():
    args = parse_args()

    # Step 1: Get all image paths
    image_paths = sorted([os.path.join(args.image_path, img) for img in os.listdir(args.image_path)])

    # Step 2: Parallel Processing
    id_list, muscle_width_list, muscle_depth_list, fat_depth_list, marbling_percentage_list, conversion_factor_list, area_px_list, area_mm_list = [], [], [], [], [], [], [], []
    canadian_classified_standard_list, lean_mask_list, outlier_list, colour_outlier_list = [], [], [], []

    max_workers = min(4, os.cpu_count() // 2)
    model = YOLO(args.model_path)
    color_model = YOLO(args.color_model_path)
    os.makedirs(f'{args.output_path}', exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_image, model, img_path, args, color_model): img_path for img_path in image_paths}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            img_id, muscle_width, muscle_depth, fat_depth, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, colour_outlier, image_path,area_mm = result
            id_list.append(img_id)
            muscle_width_list.append(muscle_width)
            muscle_depth_list.append(muscle_depth)
            fat_depth_list.append(fat_depth)
            marbling_percentage_list.append(marbling_percentage)
            canadian_classified_standard_list.append(canadian_classified_standard)
            lean_mask_list.append(lean_mask)
            conversion_factor_list.append(conversion_factor)
            area_px_list.append(area_px)
            outlier_list.append(outlier)
            colour_outlier_list.append(colour_outlier)
            area_mm_list.append(area_mm)
            if outlier == "Y" or colour_outlier == "Y":
                os.makedirs(f'{args.output_path}/outlier', exist_ok=True)
                image_outlier = cv2.imread(f"{image_path}")
                cv2.imwrite(f"{args.output_path}/outlier/{img_id}.jpg", image_outlier)


            
    # Step 3: Save and display results
    save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, args.output_path+'measurement.csv', conversion_factor_list, area_px_list, outlier_list,area_mm)
    save_marbling_csv(id_list, marbling_percentage_list, args.output_path+'marbling.csv')
    save_colouring_csv(id_list, canadian_classified_standard_list, lean_mask_list, args.output_path+'colouring.csv', colour_outlier_list)
    print_table_of_measurements(args.output_path+'measurement.csv')
    print_table_of_measurements(args.output_path+'marbling.csv')
    print_table_of_measurements(args.output_path+'colouring.csv')

if __name__ == "__main__":
    main()