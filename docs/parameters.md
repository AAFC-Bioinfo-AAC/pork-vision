# Parameters Documentation

This document provides a centralized and comprehensive reference for all functions and parameters used in the PorkVision source code. The parameters are categorized into sections based on their roles in the process for ease of navigation.

---

## Table of Contents 

1. [Helper Function Parameters](#1-helper-function-parameters)  
    1.1 [Calculations](#11-calculations)  
    1.2 [Correctors](#12-correctors)  
    1.3 [Ellipse Fitting and Plotting](#13-ellipse-fitting-and-plotting)  
    1.4 [Line, Contour, and Image Manipulations](#14-lines-and-contours)  
    1.5 [Rotation and Orientation](#15-rotation-and-orientation)  
    1.6 [Visualizations](#16-visualizations)  
2. [Inference Parameters](#2-inference-parameters)  
3. [Main Script Parameters](#3-main-script-parameters)  
4. [Results Handling Parameters](#4-results-handling-parameters)

---

## 1. Helper Function Parameters

### 1.1 Calculations

| Function              | Parameter               | Description                                                                                     | Expected Value                        |
|-----------------------|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------|
| `return_measurements`             | `depth_1`               | Starting point for depth measurement.                                                          | Tuple of two integers (x, y)           |
| `return_measurements`             | `depth_2`               | Ending point for depth measurement.                                                            | Tuple of two integers (x, y)           |
| `return_measurements`             | `width_1`               | Starting point for width measurement.                                                          | Tuple of two integers (x, y)           |
| `return_measurements`             | `width_2`               | Ending point for width measurement.                                                            | Tuple of two integers (x, y)           |
| `calculcate_midpoint_muscle_box`  | `muscle_bbox`           | Bounding box coordinates for the muscle.                                                       | `numpy.ndarray`                        |
| `calculcate_midpoint_muscle_box`  | `fat_bbox`              | Bounding box coordinates for the fat layer.                                                    | `numpy.ndarray`                        |
| `calculcate_midpoint_muscle_box`  | `orientation`           | Orientation of the fat layer relative to the muscle.                                           | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `return_min_max_mask_coords` | `contours`          | Contours from a binary mask for calculating minima and maxima.                                 | `numpy.ndarray`                        |
| `line_to_fat_box_method`          | `mid_pt_muscle`         | Midpoint of the muscle bounding box.                                                           | Tuple of two integers (x, y)           |
| `line_to_fat_box_method`          | `fat_connect`           | Connection point to the fat layer bounding box.                                                | Tuple of two integers (x, y)           |
| `line_to_fat_box_method`          | `fat_mask`              | Binary mask of the fat layer.                                                                  | `numpy.ndarray`                        |
| `convert_back_to_xyxy`            | `bbox`                  | Bounding box in polygon format to convert back to YOLO's `xyxy` format.                        | `numpy.ndarray`                        |
| `bbox_reformatter`        | `bbox`                  | Bounding box to be reformatted for compatibility with different software stacks.               | `numpy.ndarray`                        |

### 1.2 Correctors

| Function              | Parameter               | Description                                                                                     | Expected Value                        |
|-----------------------|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------|
| `correct_measurements`               | `image_line`                       | Image on which measurements are drawn.                                                         | `numpy.ndarray` (OpenCV image)         |
| `correct_measurements`               | `orientation`                      | Orientation of the fat layer relative to the muscle.                                           | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `correct_measurements`               | `rotated_fat_box`                  | Rotated bounding box for the fat layer.                                                        | `numpy.ndarray`                        |
| `correct_measurements`               | `rotated_muscle_box`               | Rotated bounding box for the muscle layer.                                                     | `numpy.ndarray`                        |
| `correct_measurements`               | `rotated_fat_mask`                 | Binary mask of the rotated fat layer.                                                          | `numpy.ndarray`                        |
| `correct_measurements`               | `p2`                               | Point where the muscle connects to the fat layer.                                              | Tuple of two integers (x, y)           |
| `correct_measurements`               | `muscle_bbox`                      | Bounding box coordinates for the muscle.                                                       | `numpy.ndarray`                        |
| `correct_measurements`               | `fat_bbox`                         | Bounding box coordinates for the fat layer.                                                    | `numpy.ndarray`                        |
| `correct_measurements`               | `angle`                            | Rotation angle derived from ellipse fitting.                                                   | Float                                   |
| `correct_measurements`               | `center`                           | Center point for rotation.                                                                     | Tuple of two integers (x, y)           |
| `distance_corrector`                 | `img`                              | Image object used for adjusting distance calculations.                                          | `numpy.ndarray` (OpenCV image)         |
| `distance_corrector`                 | `orientation`                      | Orientation of the fat layer relative to the muscle.                                           | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `distance_corrector`                 | `rotated_fat_box`                  | Rotated bounding box for the fat layer.                                                        | `numpy.ndarray`                        |
| `distance_corrector`                 | `rotated_muscle_box`               | Rotated bounding box for the muscle layer.                                                     | `numpy.ndarray`                        |
| `distance_corrector`                 | `rotated_muscle_mask`              | Binary mask of the rotated muscle layer.                                                       | `numpy.ndarray`                        |
| `distance_corrector`                 | `point_where_muscle_connects_to_fat` | Initial point of muscle-fat connection for correction.                                         | Tuple of two integers (x, y)           |
| `distance_corrector`                 | `angle`                            | Rotation angle derived from ellipse fitting.                                                   | Float                                   |
| `distance_corrector`                 | `center`                           | Center point for rotation.                                                                     | Tuple of two integers (x, y)           |
| `rotation_detector_by_angle_corrector` | `current_fat_placement`           | Current orientation of the fat layer relative to the muscle.                                   | String (`"FAT_TOP"`, `"FAT_BOTTOM"`)   |
| `rotation_detector_by_angle_corrector` | `angle`                           | Rotation angle derived from ellipse fitting.                                                   | Float                                   |
| `rotation_detector_by_angle_corrector` | `fat_bbox`                        | Bounding box coordinates for the fat layer.                                                    | `numpy.ndarray`                        |
| `rotation_detector_by_angle_corrector` | `muscle_bbox`                     | Bounding box coordinates for the muscle.                                                       | `numpy.ndarray`                        |
| `rotation_detector_by_angle_corrector` | `p2`                              | Point of interest for adjustments during rotation detection.                                   | Tuple of two integers (x, y)           |
| `line_to_fat_corrector`              | `new_p1`                           | Starting point of the line to correct.                                                         | Tuple of two integers (x, y)           |
| `line_to_fat_corrector`              | `endpt_x`                          | X-coordinate of the extended line endpoint.                                                    | Integer                                 |
| `line_to_fat_corrector`              | `endpt_y`                          | Y-coordinate of the extended line endpoint.                                                    | Integer                                 |
| `line_to_fat_corrector`              | `rotated_fat_mask`                 | Binary mask of the rotated fat layer.                                                          | `numpy.ndarray`                        |

### 1.3 Ellipse Fitting and Plotting

| Function              | Parameter               | Description                                                                                     | Expected Value                        |
|-----------------------|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------|
| `fit_ellipse_v1`          | `cnt_points_arr`        | Array of contour points for fitting the ellipse.                                                | `numpy.ndarray`                       |
| `fit_ellipse_v2`          | `cnt_points`           | Contour points to compute ellipse parameters (reshaped internally).                             | List of tuples or `numpy.ndarray`     |
| `plot_ellipse`            | `cnt_points_arr`        | Array of contour points to plot the ellipse.                                                   | `numpy.ndarray`                       |
| `plot_ellipse`            | `alphas`               | Coefficients of the fitted ellipse equation. Optional.                                          | `numpy.ndarray` (default: empty)      |
| `plot_ellipse`            | `img_show`             | Whether to display the image plot.                                                             | `boolean` (`True` or `False`)         |
| `create_fitting_ellipse` | `img`                   | Image object (used as background for overlay).                                                 | `numpy.ndarray` (OpenCV image)         |
| `create_fitting_ellipse` | `mask`                  | Binary muscle mask to fit the ellipse.                                                         | `numpy.ndarray`                        |
| `create_fitting_ellipse` | `cnt_points`            | Contour points of the muscle mask.                                                             | List of tuples or `numpy.ndarray`      |

### 1.4 Lines and Contours

| Function              | Parameter               | Description                                                                                     | Expected Value                        |
|-----------------------|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------|
| `line_extender`          | `p1`                    | Starting point of the line to extend.                                                          | Tuple of two integers (x, y)           |
| `line_extender`          | `p2`                    | Ending point of the line to extend.                                                            | Tuple of two integers (x, y)           |
| `drawlines`                     | `contour_points`        | Contour points of the binary mask to determine lines and area.                                 | `numpy.ndarray`                        |
| `drawlines`                     | `mask`                  | Binary mask where lines are drawn.                                                             | `numpy.ndarray`                        |
| `convert_contours_to_image` | `contours`        | Contours of an object to be converted into a binary mask.                                       | `numpy.ndarray`                        |
| `convert_contours_to_image` | `orig_shape`      | Original shape of the image for resizing the binary mask.                                       | Tuple of two integers (height, width)  |
| `find_nearest_contour_point`         | `contour`                          | Contour points to search for the nearest point.                                                | `numpy.ndarray`                        |
| `find_nearest_contour_point`         | `p2`                               | Reference point to find the nearest contour point.                                             | Tuple of two integers (x, y)           |
| `check_mask_presence` | `current_image`         | YOLOv8 result object for the current image.                                                    | YOLOv8 `Result` object                 |
| `mask_selector`       | `current_image`         | YOLOv8 result object containing detected masks and bounding boxes.                             | YOLOv8 `Result` object                 |
| `line_to_fat`                   | `orientation`           | Orientation of fat relative to the muscle layer.                                               | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `line_to_fat`                   | `angle`                 | Rotation angle derived from ellipse fitting.                                                   | Float                                   |
| `line_to_fat`                   | `min_h_muscle`          | Minimum horizontal point on the muscle layer.                                                  | Tuple of two integers (x, y)           |
| `line_to_fat`                   | `max_h_muscle`          | Maximum horizontal point on the muscle layer.                                                  | Tuple of two integers (x, y)           |
| `line_to_fat`                   | `min_v_muscle`          | Minimum vertical point on the muscle layer.                                                    | Tuple of two integers (x, y)           |
| `line_to_fat`                   | `max_v_muscle`          | Maximum vertical point on the muscle layer.                                                    | Tuple of two integers (x, y)           |
| `line_to_fat`                   | `rotated_fat_mask`      | Binary mask of the rotated fat layer.                                                          | `numpy.ndarray`                        |
| `box_line`                        | `muscle_bbox`           | Bounding box coordinates for the muscle.                                                       | `numpy.ndarray`                        |
| `box_line`                        | `fat_bbox`              | Bounding box coordinates for the fat layer.                                                    | `numpy.ndarray`                        |
| `box_line`                        | `img_rect`              | Image with rectangle overlays.                                                                 | `numpy.ndarray` (OpenCV image)         |
| `box_line`                        | `orientation`           | Orientation of the fat layer relative to the muscle.                                           | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `box_line_with_offset`            | `muscle_bbox`           | Bounding box coordinates for the muscle.                                                       | `numpy.ndarray`                        |
| `box_line_with_offset`            | `fat_bbox`              | Bounding box coordinates for the fat layer.                                                    | `numpy.ndarray`                        |
| `box_line_with_offset`            | `img_rect`              | Image with rectangle overlays.                                                                 | `numpy.ndarray` (OpenCV image)         |
| `box_line_with_offset`            | `orientation`           | Orientation of the fat layer relative to the muscle.                                           | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |

### 1.5 Rotation and Orientation

| Function              | Parameter               | Description                                                                                     | Expected Value                        |
|-----------------------|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------|
| `rotation_detector`      | `image_aspectratio`     | Aspect ratio of the image for determining orientation.                                         | Tuple of two integers (width, height)  |
| `rotation_detector`      | `muscle_contour`        | Contour of the muscle mask for detecting relative positions.                                   | `numpy.ndarray`                        |
| `rotation_detector`      | `fat_contour`           | Contour of the fat mask for detecting relative positions.                                      | `numpy.ndarray`                        |
| `rotation_detector_by_angle`    | `current_fat_placement` | Current orientation of the fat layer relative to the muscle.                                   | String (`"FAT_TOP"`, `"FAT_BOTTOM"`)   |
| `rotation_detector_by_angle`    | `angle`                 | Rotation angle derived from the ellipse fitting function.                                       | Float (degrees)                        |
| `rotation_detector_by_angle`    | `min_h_muscle`          | Minimum horizontal point on the muscle layer.                                                  | Tuple of two integers (x, y)           |
| `rotation_detector_by_angle`    | `max_h_muscle`          | Maximum horizontal point on the muscle layer.                                                  | Tuple of two integers (x, y)           |
| `rotation_detector_by_angle`    | `min_v_muscle`          | Minimum vertical point on the muscle layer.                                                    | Tuple of two integers (x, y)           |
| `rotation_detector_by_angle`    | `max_v_muscle`          | Maximum vertical point on the muscle layer.                                                    | Tuple of two integers (x, y)           |
| `rotate_box_line`                 | `mid_pt_muscle`         | Midpoint of the muscle bounding box.                                                           | Tuple of two integers (x, y)           |
| `rotate_box_line`                 | `fat_connect`           | Connection point to the fat layer bounding box.                                                | Tuple of two integers (x, y)           |
| `rotate_box_line`                 | `angle`                 | Rotation angle derived from ellipse fitting.                                                   | Float                                   |
| `rotate_box_line`                 | `center`                | Center point for rotation.                                                                     | Tuple of two integers (x, y)           |
| `reverse_orientation`     | `orientation`           | Current orientation of fat relative to muscle.                                                 | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `rotate_image`            | `img`                   | Image to be rotated.                                                                            | `numpy.ndarray` (OpenCV image)         |
| `rotate_image`            | `bbox`                  | Bounding box of the object to be rotated.                                                      | `numpy.ndarray`                        |
| `rotate_image`                  | `angle`                 | Rotation angle in degrees.                                                                     | Float                                   |
| `rotate_image`                  | `center`                | Center point for rotation.                                                                     | Tuple of two integers (x, y)           |

### 1.6 Visualizations

| Function              | Parameter               | Description                                                                                     | Expected Value                        |
|-----------------------|-------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------|
| `plot_polygon`            | `mask_binary`          | Binary mask to overlay the polygon.                                                            | `numpy.ndarray`                       |
| `plot_polygon`            | `contour_points`       | Contour points representing the polygon to be drawn.                                           | List of tuples or `numpy.ndarray`     |
| `plot_polygon`            | `color`                | Color of the polygon in BGR format.                                                            | Tuple of 3 integers (e.g., `(255,0,0)`) |
| `plot_polygon`            | `thickness`            | Thickness of the polygon boundary.                                                             | Integer (e.g., `1`, `2`, `3`)         |
| `draw_rotated_boxes_lines`| `img`                  | Image object on which bounding boxes and lines will be drawn.                                  | `numpy.ndarray` (OpenCV image)         |
| `draw_rotated_boxes_lines`           | `muscle_bbox`                      | Bounding box coordinates for the muscle.                                                       | `numpy.ndarray`                        |
| `draw_rotated_boxes_lines`           | `fat_bbox`                         | Bounding box coordinates for the fat layer.                                                    | `numpy.ndarray`                        |
| `draw_rotated_boxes_lines`           | `mid_pt_muscle`                    | Midpoint of the muscle bounding box.                                                           | Tuple of two integers (x, y)           |
| `draw_rotated_boxes_lines`           | `fat_connect`                      | Connection point to the fat layer bounding box.                                                | Tuple of two integers (x, y)           |

---

## 2. Inference Parameters

This section describes the parameters used during the inference step in the code. The inference process involves running a trained YOLOv8 model on the raw images to generate predictions, including bounding boxes and segmentation masks.

| Parameter    | Description                                                   | Expected Value       |
|--------------|---------------------------------------------------------------|----------------------|
| `model_path` | Path to the trained YOLOv8 model file.                        | String (`'last.pt'`) |
| `image_path` | Path to the directory containing raw images.                  | String (`'raw_images/'`) |
| `save`       | Whether to save the results of inference.                     | Boolean (`True` or `False`) |

---

## 3. Main Script Parameters

This section outlines the parameters passed to the functions within the main script. The main script processes YOLOv8 inference results, applies geometric and rotation-based transformations, and calculates measurements such as muscle depth, width, and the distance between muscle and fat layers.

| Function                           | Parameter                | Description                                                                                     | Expected Value                         |
|------------------------------------|--------------------------|-------------------------------------------------------------------------------------------------|----------------------------------------|
| `mask_selector`                    | `result`                | YOLOv8 inference result containing detected masks and bounding boxes.                          | YOLOv8 `Result` object                 |
| `convert_contours_to_image`        | `muscle_cnt`            | Contour points for the muscle mask to be converted into a binary mask.                         | `numpy.ndarray`                        |
| `convert_contours_to_image`        | `orig_shape`            | Original shape of the image for resizing the binary mask.                                       | Tuple of two integers (height, width)  |
| `create_fitting_ellipse`           | `img`                   | Original image used as a background for the ellipse overlay.                                   | `numpy.ndarray` (OpenCV image)         |
| `create_fitting_ellipse`           | `muscle_mask`           | Binary mask of the muscle region used to fit the ellipse.                                       | `numpy.ndarray`                        |
| `create_fitting_ellipse`           | `muscle_cnt`            | Contour points of the muscle mask used to fit the ellipse.                                      | `numpy.ndarray`                        |
| `rotation_detector`                | `orig_shape`            | Original dimensions of the image for determining fat-to-muscle orientation.                    | Tuple of two integers (width, height)  |
| `rotation_detector`                | `msc_contour`           | Contour points of the muscle region for detecting relative orientation.                        | `numpy.ndarray`                        |
| `rotation_detector`                | `fat_contour`           | Contour points of the fat region for detecting relative orientation.                           | `numpy.ndarray`                        |
| `rotate_image`                     | `img`                   | Image to be rotated based on ellipse fitting results.                                           | `numpy.ndarray` (OpenCV image)         |
| `rotate_image`                     | `bbox`                  | Bounding box to be rotated alongside the image.                                                | `numpy.ndarray`                        |
| `rotate_image`                     | `angle`                 | Rotation angle for transforming the image and bounding box.                                     | Float                                   |
| `rotate_image`                     | `center`                | Center point for rotation.                                                                     | Tuple of two integers (x, y)           |
| `drawlines`                        | `contours`              | Contour points of the muscle mask for generating measurement lines.                            | `numpy.ndarray`                        |
| `drawlines`                        | `img_rotated`           | Rotated image on which measurement lines are drawn.                                            | `numpy.ndarray` (OpenCV image)         |
| `line_to_fat`                      | `orientation`           | Orientation of the fat layer relative to the muscle.                                           | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `line_to_fat`                      | `angle`                 | Rotation angle derived from the ellipse fitting function.                                       | Float                                   |
| `line_to_fat`                      | `min_h_muscle`          | Minimum horizontal point on the muscle layer.                                                  | Tuple of two integers (x, y)           |
| `line_to_fat`                      | `max_h_muscle`          | Maximum horizontal point on the muscle layer.                                                  | Tuple of two integers (x, y)           |
| `line_to_fat`                      | `min_v_muscle`          | Minimum vertical point on the muscle layer.                                                    | Tuple of two integers (x, y)           |
| `line_to_fat`                      | `max_v_muscle`          | Maximum vertical point on the muscle layer.                                                    | Tuple of two integers (x, y)           |
| `line_to_fat`                      | `rotated_fat_mask`      | Rotated binary mask of the fat layer for measuring its alignment to the muscle.                | `numpy.ndarray`                        |
| `correct_measurements`             | `img_line`              | Annotated image with drawn measurement lines.                                                  | `numpy.ndarray` (OpenCV image)         |
| `correct_measurements`             | `orientation`           | Orientation of the fat layer relative to the muscle.                                           | String (`"FAT_TOP"`, `"FAT_RIGHT"`)    |
| `correct_measurements`             | `rotated_fat_box`       | Rotated bounding box for the fat layer.                                                        | `numpy.ndarray`                        |
| `correct_measurements`             | `rotated_muscle_box`    | Rotated bounding box for the muscle layer.                                                     | `numpy.ndarray`                        |
| `correct_measurements`             | `rotated_fat_mask`      | Rotated binary mask for the fat layer.                                                         | `numpy.ndarray`                        |
| `correct_measurements`             | `p2`                    | Ending point of the measurement line connecting muscle and fat layers.                         | Tuple of two integers (x, y)           |
| `correct_measurements`             | `muscle_bbox`           | Bounding box coordinates for the muscle layer.                                                 | `numpy.ndarray`                        |
| `correct_measurements`             | `fat_bbox`              | Bounding box coordinates for the fat layer.                                                    | `numpy.ndarray`                        |
| `correct_measurements`             | `angle`                 | Rotation angle derived from ellipse fitting.                                                   | Float                                   |
| `correct_measurements`             | `center`                | Center point for rotation.                                                                     | Tuple of two integers (x, y)           |

---

## 4. Results Handling Parameters

This section describes the parameters used for processing and exporting the results of the analysis. The results handling step involves creating a table of measurements, converting pixel-based measurements to millimeters, and exporting the results as a CSV file for further analysis or reporting.

| Function                  | Parameter                | Description                                                                                     | Expected Value                        |
|---------------------------|--------------------------|-------------------------------------------------------------------------------------------------|---------------------------------------|
| `pd.DataFrame`            | `list_of_measurements`   | Zipped list containing image IDs and corresponding measurements (depth, width, etc.).          | List of tuples                        |
| `pd.DataFrame`            | `columns`               | Column names for the DataFrame.                                                                | List of Strings                       |
| `pd.concat`               | `df.iloc[:,1:5] / 140`  | Conversion of pixel-based measurements to millimeters (1 mm ≈ 140 pixels).                     | `pandas.DataFrame`                    |
| `pd.rename`               | `columns`               | Renames columns in the DataFrame to indicate units (e.g., `_px` to `_mm`).                     | Dictionary mapping old to new names   |
| `pd.to_csv`               | `path`                  | File path to save the results table in CSV format.                                             | String (e.g., `'runs/results.csv'`)   |

---