from utils.imports import *
import os, time, cv2, numpy as np
from ultralytics import YOLO
from utils.preprocess import mask_selector, append_None_values_to_measurement_lists, convert_contours_to_image
from utils.orientation import orient_muscle_and_fat_using_adjacency
from utils.measurement import measure_longest_horizontal_segment, find_midline_using_fat_extremes, measure_vertical_segment, extend_vertical_line_to_fat, get_muscle_rotation_angle, measure_ruler
from utils.colouring import create_coloring_standards
from utils.postprocess import save_annotated_image, save_to_roi, extract_image_id
from utils.timer import time_program

model = None
color_model = None

def init_models(model_path, color_model_path):
    global model, color_model
    model       = YOLO(model_path)
    color_model = YOLO(color_model_path)

def _ensure_models_ready():
    if model is None or color_model is None:
        raise RuntimeError(
            "YOLO models are not initialized. Call init_models() in this module "
            "or use ProcessPoolExecutor(initializer=utils.processImage.init_models, ...)."
        )

def process_image(image_path, args):
    """Process an image: Run YOLO inference, extract measurements, and save annotated output."""
    try:
        _ensure_models_ready()
        start_time = time.time()
        outlier = None
        color_outlier = None
        image_id = extract_image_id(image_path)
        debug_messages = []
        debug_messages.append(f"{image_id}:")
        print("\nProcessing:", image_id)

        run_measurement = 'measurement' in args.outputs
        run_marbling   = 'marbling' in args.outputs or 'colour' in args.outputs
        run_colour     = 'colour' in args.outputs
        
        # Step 1: YOLO Inference
        debug_messages.append("SEGMENTATION MODEL:")
        model_start = time.time() 
        results = model(image_path, save=False, retina_masks=True)[0]  # This disables automatic saving into subfolders
        minutes, seconds = time_program(model_start)
        debug_messages.append(f"Segmentation Model Time: {minutes}:{seconds:02d}\n\n")
        
        # Save the result manually to the 'predict' folder
        if args.minimal == False:
            os.makedirs(f'{args.output_path}/predict', exist_ok=True)
            save_path = f'{args.output_path}/predict/{extract_image_id(image_path)}.jpg'
            results.save(save_path)  # Save the annotated image to the specified path
            debug_messages.append(f"Image is saved to {args.output_path}predict\n\n")

        # Step 2: Preprocessing
        preprocessing_start = time.time()
        debug_messages.append("PREPROCESSING:")
        muscle_bbox, muscle_mask, fat_bbox, fat_mask, debug_messages = mask_selector(results, debug_messages)
        if muscle_bbox is None or fat_bbox is None:
            outlier = "Y"
            debug_messages.append(f"ERROR: Did not detect a muscle/fat bounding box.")
            return extract_image_id(image_path), 0, 0, 0, 0, 0, 0, 0, 0, outlier, color_outlier, image_path, 0, debug_messages
        debug_messages.append("Converting muscle mask to image")
        muscle_binary_mask, debug_messages = convert_contours_to_image(muscle_mask, results.orig_shape, debug_messages)
        debug_messages.append("Converting fat mask to image")
        fat_binary_mask, debug_messages = convert_contours_to_image(fat_mask, results.orig_shape, debug_messages)
        minutes,seconds = time_program(preprocessing_start)
        debug_messages.append("Preprocessing Finished")
        debug_messages.append(f"Preprocessing Time: {minutes}:{seconds:02d}\n\n")
        
        # Step 3: Orientation
        orientation_start = time.time()
        debug_messages.append("ORIENTATION:")
        rotated_image, rotated_muscle_mask, rotated_fat_mask, final_angle, outlier, debug_messages = orient_muscle_and_fat_using_adjacency(
            results.orig_img, muscle_binary_mask, fat_binary_mask, outlier, debug_messages
        )
        debug_messages.append("Orientation Finished")
        minutes,seconds = time_program(orientation_start)
        debug_messages.append(f"Orientation Time: {minutes}:{seconds:02d}\n\n")

        # Step 4: Conversion Factor Calculation
        debug_messages.append(f"CONVERSION FACTOR:")
        conversion_calculation_time = time.time()
        debug_messages.append("Measuring Ruler in pixels to determine conversion factor")
        conversion_factor, outlier, debug_messages = measure_ruler(rotated_image, image_id, outlier, args.minimal, debug_messages)
        if conversion_factor == None:
            outlier = "Y"
            debug_messages.append(f"ERROR: Conversion Factor calculation, using default.")
            conversion_factor = 10/140 #mm/px
        else:
            debug_messages.append(f"Success!")
            debug_messages.append(f"Conversion factor (mm/px) = {conversion_factor}")
        minutes,seconds = time_program(conversion_calculation_time)
        debug_messages.append(f"Conversion Factor Finished")
        debug_messages.append(f"Conversion Calculation Time: {minutes}:{seconds:02d}\n\n")

        # Step 5: Create Canadian Standard chart using A.I model.
        if run_marbling or run_measurement:
            debug_messages.append("COLOR MODEL:")
            color_model_start = time.time()
            debug_messages.append("Creating color standards using YOLO model.")
            canadian_standards, outlier, debug_messages = create_coloring_standards(rotated_image, color_model, image_id, args.output_path+'/colouring', outlier, args.minimal, debug_messages)
            if len(canadian_standards) == 7:
                debug_messages.append("Standards successfully created (RGB)\n"
                                    f"C6: {canadian_standards[0]}\n"
                                    f"C5: {canadian_standards[1]}\n"
                                    f"C4: {canadian_standards[2]}\n"
                                    f"C3: {canadian_standards[3]}\n"
                                    f"C2: {canadian_standards[4]}\n"
                                    f"C1: {canadian_standards[5]}\n"
                                    f"C0: {canadian_standards[6]}\n"
                )
            else:
                debug_messages.append("Error with finding standards")
            minutes, seconds = time_program(color_model_start)
            debug_messages.append(f"Color Model Finished")
            debug_messages.append(f"Color Model Time: {minutes}:{seconds:02d}\n\n")
        else:
            canadian_standards = None
            debug_messages.append(f"Skipping creation of color standards\n\n")

        # Step 6: Find Marbling
        if run_marbling or run_measurement:
            debug_messages.append("MARBLING:")
            mr_folder = os.path.join(args.output_path, 'marbling', 'regions')
            os.makedirs(mr_folder, exist_ok=True)

            region_path = os.path.join(mr_folder, f"{image_id}_crop.png")
            roi_path    = os.path.join(mr_folder, f"{image_id}_roi.png")

            roi_mask = rotated_muscle_mask.astype(np.uint8)
            area_px   = int(np.count_nonzero(roi_mask))
            area_mm   = round(area_px * (conversion_factor ** 2), 2)

            colour_crop = cv2.bitwise_and(rotated_image, rotated_image, mask=roi_mask)
            cv2.imwrite(region_path, colour_crop)                  
            cv2.imwrite(roi_path,    roi_mask)                      

            debug_messages.append(f"Wrote crop  -> {region_path}")
            debug_messages.append(f"Wrote ROI   -> {roi_path}")
        else:
            marbling_mask = eroded_mask = None
            marbling_percentage = area_px = area_mm = 0
            debug_messages.append(f"Skipping marbling analysis\n\n")

        # Step 7: Perform color grading
        if run_colour:
            debug_messages.append("COLOUR EXPORT:")
            cg_folder = os.path.join(args.output_path, 'colouring', 'regions')
            os.makedirs(cg_folder, exist_ok=True)

            colour_crop_path = os.path.join(cg_folder, f"{image_id}_colour.png")
            colour_roi_path  = os.path.join(cg_folder, f"{image_id}_roi.png")

            # reuse roi_mask + colour_crop produced in Step 6
            cv2.imwrite(colour_crop_path, colour_crop)
            cv2.imwrite(colour_roi_path,  roi_mask)

            green_path = os.path.join(cg_folder, f"{image_id}_G.png")
            green_channel = colour_crop[:, :, 1].copy()        # OpenCV: [:, :, 1] is G
            cv2.imwrite(green_path, green_channel)

            std_path   = os.path.join(cg_folder, f"{image_id}_std.txt")
            np.savetxt(std_path, canadian_standards, fmt='%d', delimiter=',')   # 7 rows, “R,G,B”

            debug_messages.append(f"Wrote colour crop  -> {colour_crop_path}")
            debug_messages.append(f"Wrote colour ROI   -> {colour_roi_path}")
        else:
            canadian_classified_standard = lean_mask = color_outlier = None
            debug_messages.append(f"Skipping color grading\n\n")

        # Step 8: Measurement
        if run_measurement:
            debug_messages.append("MEASUREMENT:")
            measurement_start = time.time()
            debug_messages.append("Finding the muscle angle")
            angle = get_muscle_rotation_angle(rotated_muscle_mask)
            if angle is None:
                outlier = "Y"
                debug_messages.append(f"ERROR: ANGLE is None")
                return extract_image_id(image_path), 0, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm, debug_messages
            debug_messages.append("Success!")
            debug_messages.append("Finding the longest horizontal segment in the muscle mask")
            muscle_width_start, muscle_width_end = measure_longest_horizontal_segment(rotated_muscle_mask, angle)
            if muscle_width_start is None or muscle_width_end is None:
                outlier = "Y"
                debug_messages.append(f"ERROR: Muscle width is None")
                return extract_image_id(image_path), 0, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm, debug_messages
            debug_messages.append("Success!")
            muscle_width = np.linalg.norm(np.array(muscle_width_start) - np.array(muscle_width_end))
            debug_messages.append(f"Muscle Width in pixels = {muscle_width}")
            debug_messages.append(f"Finding the midline between muscle and best fat")
            midline_position, midline_point = find_midline_using_fat_extremes(rotated_fat_mask)
            if midline_position is None:
                outlier = "Y"
                debug_messages.append(f"ERROR: Midline position is None")
                return extract_image_id(image_path), muscle_width, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm, debug_messages
            debug_messages.append("Success!")
            debug_messages.append("Finding the Muscle depth")
            muscle_depth_start, muscle_depth_end = measure_vertical_segment(rotated_muscle_mask, midline_position, angle, (1/(conversion_factor)*10))
            if muscle_depth_start is None or muscle_depth_end is None:
                outlier = "Y"
                debug_messages.append(f"ERROR: Muscle Depth is None")
                return extract_image_id(image_path), muscle_width, 0, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm, debug_messages
            debug_messages.append("Success!")
            muscle_depth = np.linalg.norm(np.array(muscle_depth_start) - np.array(muscle_depth_end))
            debug_messages.append(f"Muscle Depth in pixels = {muscle_depth} px")
            debug_messages.append("Finding the fat depth!")
            fat_depth_start, fat_depth_end = extend_vertical_line_to_fat(rotated_fat_mask, (muscle_depth_start, muscle_depth_end))
            if fat_depth_start is None or fat_depth_end is None:
                outlier = "Y"
                debug_messages.append(f"ERROR: Fat depth is None")
                return extract_image_id(image_path), muscle_width, muscle_depth, 0, marbling_percentage, canadian_classified_standard, lean_mask, conversion_factor, area_px, outlier, color_outlier, image_path, area_mm, debug_messages
            debug_messages.append(f"Success!")
            fat_depth = np.linalg.norm(np.array(fat_depth_start) - np.array(fat_depth_end))
            debug_messages.append(f"Fat depth in pixels = {fat_depth}")
            minutes,seconds = time_program(measurement_start)
            debug_messages.append("MEASUREMENT FINISHED")
            debug_messages.append(f"Measurement Time: {minutes}:{seconds:02d}\n\n")
        else:
            muscle_width = muscle_depth = fat_depth = 0
            debug_messages.append(f"Skipping measurement\n\n")

        # Step 9: Save annotated image
        if run_measurement and args.minimal == False:
            debug_messages.append("Saving the annotated image")
            save_annotated_image(
                rotated_image, (muscle_width_start, muscle_width_end), (muscle_depth_start, muscle_depth_end),
                (fat_depth_start, fat_depth_end), image_path, args.output_path+'/annotated_images', args.minimal
            )
            debug_messages.append("Saving the ROI images")
            save_to_roi(muscle_width_start, muscle_width_end, muscle_depth_start, muscle_depth_end, fat_depth_start, fat_depth_end, image_id=extract_image_id(image_path), rois_folder=args.output_path+'/rois')
        
        minutes,seconds = time_program(start_time)
        debug_messages.append(f"OVERALL RUNTIME for {image_id}: {minutes}:{seconds:02d}")
        
        return (image_id, int(round(muscle_width)), int(round(muscle_depth)), int(round(fat_depth)), conversion_factor, area_px, area_mm, outlier, image_path, debug_messages)

    except Exception as e:
        debug_messages.append(f"Error processing {image_id}: {e}")
        minutes, seconds = time_program(start_time)
        debug_messages.append(f"OVERALL RUNTIME for {image_id}: {minutes}:{seconds:02d}")

        return (image_id, 0, 0, 0, 0.0, 0, 0, "Y", image_path, debug_messages)