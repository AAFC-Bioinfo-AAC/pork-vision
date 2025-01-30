# main.py

import argparse
import os
import math
import cv2
import pandas as pd
from tabulate import tabulate
from csv import reader
from skimage import draw
from ultralytics import YOLO
from utils.calculations import return_measurements
from utils.ellipses import create_fitting_ellipse
from utils.lines import drawlines, line_to_fat, mask_selector, check_mask_presence, convert_contours_to_image
from utils.rotation import rotate_image, rotation_detector, reverse_orientation
from utils.visualizations import plot_polygon
from utils.correctors import correct_measurements

def parse_args():
    parser = argparse.ArgumentParser(description="Run PorkVision Inference and Analysis")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/last.pt",
        help="Path to the YOLO model",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/raw_images/",
        help="Path to the raw images",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/segment",
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
    results = model(args.image_path, save=True, project=args.output_path)

    id_list = []
    ld_depth_list = []
    ld_width_list = []
    muscle_to_fat_list = []

    ld_depth_yellow_list = []
    ld_depth_purple_list = []
    fat_depth_yellow_list = []
    fat_depth_purple_list = []

    for i in range(len(results)):
        result = results[i]
        print(result.path)

        if not check_mask_presence(result):
            print(f"Model failed to create a mask for either fat or muscle for image {result.path}. Please calculate dimensions manually.")
            id_list.append(result.path.split("/")[-1].split("_")[0])
            ld_depth_list.append(None)
            ld_width_list.append(None)
            muscle_to_fat_list.append(None)
            continue

        reversion = False
        bad_mask = False

        orig_shape = result.orig_shape
        muscle_bbox, muscle_cnt, fat_bbox, fat_cnt = mask_selector(result)

        muscle_mask = convert_contours_to_image(muscle_cnt[0], orig_shape)
        fat_mask = convert_contours_to_image(fat_cnt[0], orig_shape)

        orig_image = result.orig_img

        center, angle, img_ellipse = create_fitting_ellipse(orig_image.copy(), muscle_mask, muscle_cnt)

        msc_contour, _ = cv2.findContours(muscle_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        fat_contour, _ = cv2.findContours(fat_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        orientation = rotation_detector(
            orig_shape,
            msc_contour[0].reshape(-1, 2),
            fat_contour[0].reshape(-1, 2),
        )
        print(orientation)

        img_rotated, _, bbox = rotate_image(orig_image.copy(), bbox=muscle_bbox, angle=angle, center=center)
        _, rotated_muscle_box, _ = rotate_image(orig_image.copy(), bbox=muscle_bbox, angle=angle, center=center)
        _, rotated_fat_box, _ = rotate_image(orig_image.copy(), bbox=fat_bbox, angle=angle, center=center)
        rotated_muscle_mask, _, _ = rotate_image(muscle_mask.copy(), bbox=muscle_bbox, angle=angle, center=center)
        rotated_fat_mask, _, _ = rotate_image(fat_mask.copy(), bbox=fat_bbox, angle=angle, center=center)

        contours, hierarchy = cv2.findContours(rotated_muscle_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        (
            area_muscle,
            min_h_muscle,
            max_h_muscle,
            min_v_muscle,
            max_v_muscle,
            image_line,
        ) = drawlines(contours[0].reshape(-1, 2), img_rotated.copy())

        orientations_to_try = [
            "FAT_TOP",
            "FAT_BOTTOM",
            "FAT_RIGHT",
            "FAT_LEFT",
        ]

        p1, p2, max_fat_pt, ld_depth, ld_width = None, None, None, None, None
        success = False

        for orient in orientations_to_try:
            try:
                p1, p2, max_fat_pt, ld_depth, ld_width = line_to_fat(
                    orient,
                    angle,
                    min_h_muscle,
                    max_h_muscle,
                    min_v_muscle,
                    max_v_muscle,
                    rotated_fat_mask.copy(),
                )

                if p1 is not None and p2 is not None and max_fat_pt is not None:
                    print(f"Success with orientation {orient}")
                    success = True
                    break
            except Exception as e:
                print(f"Failed with orientation {orient}: {e}")

        if not success:
            print(f"Critical Error: No valid orientation found for image {result.path}. Skipping.")
            id_list.append(result.path.split("/")[-1].split("_")[0])
            ld_depth_list.append(None)
            ld_width_list.append(None)
            muscle_to_fat_list.append(None)
            continue

        print(ld_width)

        image_old = image_line

        muscle_to_fat = abs(math.dist(p2, max_fat_pt))

        ld_depth_list.append(ld_depth)
        ld_width_list.append(ld_width)
        muscle_to_fat_list.append(muscle_to_fat)
        id_list.append(result.path.split("/")[-1].split("_")[0])

        print(
            "Orientation required reversion: ",
            reversion,
            "\n",
            "First fat mask was a bad mask: ",
            bad_mask,
        )

        try:
            if reversion and not bad_mask:
                print("Reverted")
                print(orientation, orientation_reverse)
                img_final, ld_depth_yellow, ld_depth_purple, fat_depth_yellow, fat_depth_purple = correct_measurements(
                    image_old.copy(),
                    orientation_reverse,
                    rotated_fat_box,
                    rotated_muscle_box,
                    rotated_fat_mask.copy(),
                    p2,
                    muscle_bbox,
                    fat_bbox,
                    angle,
                    center,
                )
            elif reversion and bad_mask:
                print("Reverted and new mask")
                print(orientation, orientation_reverse)
                img_final, ld_depth_yellow, ld_depth_purple, fat_depth_yellow, fat_depth_purple = correct_measurements(
                    image_old.copy(),
                    orientation_reverse,
                    rotated_fat_box,
                    rotated_muscle_box,
                    new_rotated_fat_mask.copy(),
                    p2,
                    muscle_bbox,
                    fat_bbox,
                    angle,
                    center,
                )
            else:
                img_final, ld_depth_yellow, ld_depth_purple, fat_depth_yellow, fat_depth_purple = correct_measurements(
                    image_old.copy(),
                    orientation,
                    rotated_fat_box,
                    rotated_muscle_box,
                    rotated_fat_mask.copy(),
                    p2,
                    muscle_bbox,
                    fat_bbox,
                    angle,
                    center,
                )
        except Exception as e:
            print(f"ERROR in measurement correction: {e}")

        id_list.append(result.path.split("/")[-1].split("_")[0])
        ld_depth_yellow_list.append(ld_depth_yellow)
        ld_depth_purple_list.append(ld_depth_purple)
        ld_width_list.append(ld_width)
        fat_depth_yellow_list.append(fat_depth_yellow)
        fat_depth_purple_list.append(fat_depth_purple)

        cv2.imwrite(
            os.path.join("output/annotated_images", f"{result.path.split('/')[-1].split('.')[0]}_annotated.JPG"),
            img_final,
        )

    df = pd.DataFrame(
        list(zip(id_list, ld_depth_yellow_list, ld_depth_purple_list, ld_width_list, fat_depth_yellow_list, fat_depth_purple_list)),
        columns=["image_id", "ld_depth_yellow_px", "ld_depth_purple_px", "ld_width_px", "fat_depth_yellow_px", "fat_depth_purple_px"],
    )

    df_mm = df.iloc[:, 1:6] / 140
    df_mm.columns = ["ld_depth_yellow_mm", "ld_depth_purple_mm", "ld_width_mm", "fat_depth_yellow_mm", "fat_depth_purple_mm"]
    df = pd.concat([df, df_mm], axis=1)

    df.to_csv(args.results_csv, index=False)

    print(tabulate(df, headers="keys", tablefmt="pipe"))

if __name__ == "__main__":
    main()