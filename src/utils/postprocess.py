from utils.imports import *
from roifile import ImagejRoi, ROI_TYPE, ROI_OPTIONS
from tabulate import tabulate

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

def save_annotated_image(image, muscle_width, muscle_depth, fat_depth, image_path, output_path):
    """
    Draws measurement lines on the image and saves it.

    Parameters:
    - image (numpy.ndarray): The rotated image.
    - muscle_width (tuple): (leftmost, rightmost) points defining the muscle width.
    - muscle_depth (tuple): (start, end) points defining the muscle depth.
    - fat_depth (tuple): (start, end) points defining the fat depth.
    - image_path (str): Original image file path (to extract filename).
    - output_path (str): Directory to save annotated images.
    """

    # Create a copy of the image for annotation
    annotated_image = image.copy()

    # Define colors (BGR format)
    width_color = (0, 255, 0)  # Green for muscle width
    depth_color = (0, 255, 255)  # Yellow for muscle depth
    fat_color   = (255, 0, 0)  # Blue for fat depth
    thickness = 5

    # Helper function to convert a point to integer coordinates.
    def to_int_point(pt):
        return (int(round(pt[0])), int(round(pt[1])))

    # Draw muscle width line
    if muscle_width:
        pt1 = to_int_point(muscle_width[0])
        pt2 = to_int_point(muscle_width[1])
        cv2.line(annotated_image, pt1, pt2, width_color, thickness)

    # Draw muscle depth line
    if muscle_depth:
        pt1 = to_int_point(muscle_depth[0])
        pt2 = to_int_point(muscle_depth[1])
        cv2.line(annotated_image, pt1, pt2, depth_color, thickness)
    else:
        print("no muscle depth")

    # Draw fat depth line
    if fat_depth:
        pt1 = to_int_point(fat_depth[0])
        pt2 = to_int_point(fat_depth[1])
        cv2.line(annotated_image, pt1, pt2, fat_color, thickness)
    else:
        print("no fat depth")

    # Extract filename and define output path
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.basename(image_path)
    output_file = os.path.join(output_path, f"annotated_{filename}")

    # Save the annotated image
    cv2.imwrite(output_file, annotated_image)

    print(f"Annotated image saved: {output_file}\n")

def save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, output_csv_path, conversion_factor_list, area_px_list):
    """
    Saves the measurement results to a CSV file.

    Parameters:
    - id_list (list): List of image IDs.
    - muscle_width_list (list): List of measured muscle widths.
    - muscle_depth_list (list): List of measured muscle depths.
    - fat_depth_list (list): List of measured fat depths.
    - output_csv_path (str): Path to save the CSV file.
    - conversion_factor_list (list): List of measured conversions factors.
    - area_px_list (list): List of measured px representing muscle region.
    """

    df = pd.DataFrame({
        "Image ID": id_list,
        "Muscle Width (px)": muscle_width_list,
        "Muscle Depth (px)": muscle_depth_list,
        "Fat Depth (px)": fat_depth_list,
    })

    df_mm = df.iloc[:, 1:].multiply(conversion_factor_list, axis=0)
    df_mm.columns = ["Muscle Width (mm)", "Muscle Depth (mm)", "Fat Depth (mm)"]

    # Concatenate pixel and mm measurements
    df = pd.concat([df, df_mm], axis=1)
    df_conversion = pd.DataFrame({
        "Conversion Factor (mm/px)": conversion_factor_list})
    df = pd.concat([df, df_conversion], axis=1)
    df_areapx = pd.DataFrame({
        "Area (px^2)": area_px_list
    })
    df = pd.concat([df,df_areapx], axis=1)
    # Save DataFrame to CSV
    df.to_csv(output_csv_path, index=False)

    #print(f"Results saved to: {output_csv_path}")

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
        print("\nResults:")
        print(tabulate(df, headers="keys", tablefmt="pipe", showindex=False))

    except Exception as e:
        print(f"Error reading results CSV: {e}")


def save_to_roi(muscle_width_start, muscle_width_end, muscle_depth_start, muscle_depth_end, fat_depth_start, fat_depth_end, image_id, rois_folder):
    '''
    Generates an ROI for each segment and saves it to file.
    inputs:
        muscle_width_start, muscle_width_end are the coordinate points of the horizontal loin segment.
        musce_deth_start, muscle_depth_end are the coordinate points of the vertical loin segment.
        fat_depth_start, fat_depth_end are the coordinate point of the fat segment.
        image_id: a string of the image ID number
        rois_folder: path to save the rois.
    '''

    def to_int_point(pt):
        return (int(round(pt[0])), int(round(pt[1])))

    horizontal_start = to_int_point(muscle_width_start)
    horizontal_end = to_int_point(muscle_width_end)
    vertical_start = to_int_point(muscle_depth_start)
    vertical_end = to_int_point(muscle_depth_end)
    fat_start = to_int_point(fat_depth_start)
    fat_end = to_int_point(fat_depth_end)
    horiz_pts = np.array([np.array(horizontal_start), np.array(horizontal_end)])
    vert_pts = np.array([np.array(vertical_start), np.array(vertical_end)])
    fat_pts = np.array([np.array(fat_start), np.array(fat_end)])
    roi_h = ImagejRoi.frompoints(horiz_pts, name=f"{image_id}_horizontal")
    roi_h.roitype = ROI_TYPE.POLYGON
    roi_h.version=227
    roi_h.options = ROI_OPTIONS(0)

    base_output_dir = os.path.join(rois_folder, image_id)
    os.makedirs(base_output_dir, exist_ok=True)

    # print("roi_h coordinates =", roi_h.coordinates())
    roi_h.tofile(os.path.join(base_output_dir, f'{image_id}_horizontal.roi'))

    roi_v = ImagejRoi.frompoints(vert_pts, name=f"{image_id}_vertical")
    roi_v.roitype = ROI_TYPE.POLYGON
    roi_v.version=227
    roi_v.options = ROI_OPTIONS(0)
    # print("roi_v coordinates =", roi_v.coordinates())
    roi_v.tofile(os.path.join(base_output_dir, f'{image_id}_vertical.roi'))

    roi_f = ImagejRoi.frompoints(fat_pts, name=f"{image_id}_fat")
    roi_f.roitype = ROI_TYPE.POLYGON
    roi_f.version=227
    roi_f.options = ROI_OPTIONS(0)
    # print("roi_f coordinates =", roi_f.coordinates())
    roi_f.tofile(os.path.join(base_output_dir, f'{image_id}_fat.roi'))