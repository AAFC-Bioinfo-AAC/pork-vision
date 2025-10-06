#!/usr/bin/env python
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food, 2025.
# Pork-vision: pork chop image analysis pipeline
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

LICENSE_NOTICE = """\

Pork-vision: pork chop image analysis pipeline. This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it under certain condition. 
For license details, see <https://www.gnu.org/licenses/>.

"""

def print_LICENSE_NOTICE() -> None:
    print(LICENSE_NOTICE)

# print on startup
print_LICENSE_NOTICE()

from utils.imports import *
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultralytics import YOLO
from utils.preprocess import mask_selector, append_None_values_to_measurement_lists, convert_contours_to_image
from utils.orientation import orient_muscle_and_fat_using_adjacency
from utils.marbling import compute_percentage, save_marbling_csv
from utils.colouring import create_coloring_standards
from utils.measurement import measure_longest_horizontal_segment, find_midline_using_fat_extremes, measure_vertical_segment, extend_vertical_line_to_fat, get_muscle_rotation_angle, measure_ruler
from utils.postprocess import save_annotated_image, save_results_to_csv, print_table_of_measurements, extract_image_id, save_to_roi
from utils.timer import time_program
from utils.processImage import process_image, init_models
import time, subprocess, textwrap, pathlib, glob, shutil

FIJI_CMD = os.environ.get("FIJI_CMD", "fiji") 

def debug_info(debug_messages,image_id,args):
    '''
    If debug=True in args, then output the debug messages collected into a debug folder
    as .txt files for each image.
    '''
    if args.debug == True:
        os.makedirs(f"{args.output_path}/debug", exist_ok=True)
        with open(f'{args.output_path}/debug/{image_id}_DEBUGINFO.txt', 'w') as file:
            for string in debug_messages:
                file.write(string+"\n")
            file.close()

def main():
    start_time = time.time()
    args = parse_args()
    run_measurement = 'measurement' in args.outputs
    run_marbling   = 'marbling'   in args.outputs or 'colour' in args.outputs
    run_colour     = 'colour'     in args.outputs

    # Step 1: collect image paths
    image_paths = sorted(
        os.path.join(args.image_path, fn)
        for fn in os.listdir(args.image_path)
        if fn.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
    )

    # Step 2: parallel YOLO + preprocess + orientation + ruler + colour-standards
    id_list, muscle_width_list, muscle_depth_list, fat_depth_list = [], [], [], []
    conversion_factor_list, area_px_list, area_mm_list = [], [], []
    outlier_list = []

    max_workers = min(16, os.cpu_count() // 2)
    os.makedirs(args.output_path, exist_ok=True)
    m, s = time_program(start_time)
    print(f"RUNTIME BEFORE IMAGE ANALYSIS STARTS: {m}:{s:02d}")

    parallel_start = time.time()
    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_models,
        initargs=(args.model_path, args.color_model_path)
    ) as executor:
        futures = [executor.submit(process_image, p, args) for p in image_paths]
        for future in as_completed(futures):
            (img_id, mw, md, fd, cf, apx, amm, outl, orig_path, debug_msgs) = future.result()

            id_list.append(img_id)
            muscle_width_list.append(mw)
            muscle_depth_list.append(md)
            fat_depth_list.append(fd)
            conversion_factor_list.append(cf)
            area_px_list.append(apx)
            area_mm_list.append(amm)
            outlier_list.append(outl)

            debug_info(debug_msgs, img_id, args)

            if outl == "Y":
                od = os.path.join(args.output_path, 'outlier')
                os.makedirs(od, exist_ok=True)
                if os.path.exists(orig_path):
                    im = cv2.imread(orig_path)
                    if im is not None:
                        cv2.imwrite(os.path.join(od, f"{img_id}.jpg"), im)

    m, s = time_program(parallel_start)
    print(f"TOTAL RUNTIME FOR IMAGE ANALYSIS: {m}:{s:02d}\n")

    post_start = time.time()

    # Step 3: measurement CSV
    if run_measurement:
        meas_csv = os.path.join(args.output_path, 'measurement.csv')
        save_results_to_csv(id_list, muscle_width_list, muscle_depth_list, fat_depth_list, meas_csv, conversion_factor_list, area_px_list, outlier_list, area_mm_list)
        print_table_of_measurements(meas_csv)

    # Step 4: batch-run Fiji macro for marbling
    if run_marbling or run_colour:
        print("BATCH RUNNING Fiji macro over all muscle regions…")
        marbling_root = os.path.join(args.output_path, 'marbling')
        regions_dir = os.path.join(args.output_path, 'marbling', 'regions')
        masks_dir   = os.path.join(args.output_path, 'marbling', 'masks')
        os.makedirs(regions_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)

        macro_path = pathlib.Path(__file__).parent / "macros" / "batch_marble.ijm"
        cmd = [FIJI_CMD, "--headless", "-batch", str(macro_path), marbling_root]

        subprocess.run(cmd, check=True)
        print("Fiji batch done via subprocess.")

        overlay_dir = os.path.join(marbling_root, 'overlays')
        os.makedirs(overlay_dir, exist_ok=True)

        marb_ids, marb_pcts = [], []
        for fn in sorted(os.listdir(masks_dir)):
            if not fn.endswith("_crop.png"):
                continue
            base = fn[:-9]
            mask_marbling = cv2.imread(os.path.join(masks_dir, fn),
                                    cv2.IMREAD_GRAYSCALE)

            color_crop = cv2.imread(os.path.join(regions_dir,
                                     f"{base}_crop.png"))
            overlay     = color_crop.copy()
            yellow_bgr  = (0, 255, 255)                
            fat_pixels  = (mask_marbling == 0)         
            overlay[fat_pixels] = yellow_bgr          
            out_path = os.path.join(overlay_dir, f"{base}_overlay.png")
            cv2.imwrite(out_path, overlay)

            roi_bin = cv2.imread(os.path.join(regions_dir,
                                            f"{base}_roi.png"),
                                cv2.IMREAD_GRAYSCALE)

            pct = compute_percentage(mask_marbling, roi_bin)
            marb_ids.append(base)
            marb_pcts.append(pct)

        marbling_csv = os.path.join(args.output_path, 'marbling.csv')
        save_marbling_csv(marb_ids, marb_pcts, marbling_csv)
        print_table_of_measurements(marbling_csv)

    # Step 5: colour CSV
    if run_colour:
        print("BATCH RUNNING Fiji macro for lean-colour grading…")
        colour_root  = os.path.join(args.output_path, 'colouring')
        regions_dir  = os.path.join(colour_root, 'regions')
        lean_dir = os.path.join(args.output_path, 'marbling', 'masks')
        results_dir  = os.path.join(colour_root, 'results')
        os.makedirs(results_dir,  exist_ok=True)

        macro_path = pathlib.Path(__file__).parent / "macros" / "batch_colour.ijm"
        cmd = [FIJI_CMD, "--headless", "-batch", str(macro_path), colour_root]

        subprocess.run(cmd, check=True)
        print("Fiji colour batch done via subprocess.")

        master_rows = []
        for xls_path in glob.glob(os.path.join(results_dir, '*_colour.xls')):
            try:
                df = pd.read_csv(xls_path, sep='\t')
                if 'image_id' not in df.columns:
                    img_id = pathlib.Path(xls_path).stem.replace('_colour', '')
                    df.insert(0, 'image_id', img_id)
                cols = ['image_id', 'Standard', 'CdnCount', 'CdnPercent']
                df = df[[c for c in cols if c in df.columns]]
                master_rows.append(df)
            except Exception as err:
                print(f"Warning: could not read {xls_path}: {err}")

        if master_rows:
            out_csv = os.path.join(args.output_path, 'colour.csv')
            pd.concat(master_rows, ignore_index=True).to_csv(out_csv, index=False)
            print_table_of_measurements(out_csv)

            for xls_path in glob.glob(os.path.join(results_dir, '*_colour.xls')):
                try:
                    os.remove(xls_path)
                except OSError:
                    pass

    m, s = time_program(post_start)
    print(f"POST PROCESSING RUNTIME: {m}:{s:02d}\n")
    m, s = time_program(start_time)
    print(f"OVERALL RUNTIME: {m}:{s:02d}")

if __name__ == "__main__":

    main() 
