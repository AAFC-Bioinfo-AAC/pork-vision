import argparse
import os
import math
import cv2
import pandas as pd
from tabulate import tabulate
from csv import reader
from ultralytics import YOLO
from utils.ellipses import create_fitting_ellipse
from utils.lines import drawlines, line_to_fat, mask_selector, check_mask_presence, convert_contours_to_image
from utils.rotation import rotate_image, rotation_detector, reverse_orientation
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

def create_table(ID, depth, width, muscle_fat, args_parser):
    list_of_measurements = list(zip(ID, depth, width, muscle_fat))
    df = pd.DataFrame(
        list_of_measurements,
        columns=["image_id", "ld_depth_px", "ld_width_px", "fat_depth_px"],
    )
    df_mm = df.iloc[:, 1:4] / 140
    df_mm.columns = ["ld_depth_mm", "ld_width_mm", "fat_depth_mm"]

    df = pd.concat([df, df_mm], axis=1)
    column_titles = [
        "image_id",
        "ld_depth_px",
        "ld_depth_mm",
        "ld_width_px",
        "ld_width_mm",
        "fat_depth_px",
        "fat_depth_mm",
    ]
    df = df.reindex(columns=column_titles)

    df.to_csv(args_parser.results_csv, index=False)

    with open(args_parser.results_csv) as f:
        csv_f = reader(f)
        print(tabulate(csv_f, headers="firstrow", tablefmt="pipe"))


def main():
    os.makedirs("output/annotated_images", exist_ok=True)
    args = parse_args()

    model = YOLO(args.model_path)
    results = model(args.image_path, save=True, project=args.output_path)

    id_list = []
    ld_depth_list = []
    ld_width_list = []
    muscle_to_fat_list = []

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

        try:
            p1, p2, max_fat_pt, ld_depth, ld_width = line_to_fat(
                orientation,
                angle,
                min_h_muscle,
                max_h_muscle,
                min_v_muscle,
                max_v_muscle,
                rotated_fat_mask.copy(),
            )
        except:
            orientation_reverse = reverse_orientation(orientation)
            reversion = True
            try:
                p1, p2, max_fat_pt, ld_depth, ld_width = line_to_fat(
                    orientation_reverse,
                    angle,
                    min_h_muscle,
                    max_h_muscle,
                    min_v_muscle,
                    max_v_muscle,
                    rotated_fat_mask.copy(),
                )
            except:
                current_mask = 2
                bad_mask = True
                while True:
                    try:
                        new_fat_contour = result.masks.xy[current_mask]
                        new_fat_bbox = result.boxes.xyxy[current_mask]
                        new_fat_mask = convert_contours_to_image(new_fat_contour, orig_shape)
                        new_rotated_fat_mask, new_rotated_fat_box, _ = rotate_image(
                            new_fat_mask.copy(),
                            bbox=new_fat_bbox,
                            angle=angle,
                            center=center,
                        )
                        p1, p2, max_fat_pt, ld_depth, ld_width = line_to_fat(
                            orientation_reverse,
                            angle,
                            min_h_muscle,
                            max_h_muscle,
                            min_v_muscle,
                            max_v_muscle,
                            new_rotated_fat_mask.copy(),
                        )
                        break
                    except:
                        current_mask += 1
                    if current_mask >= len(result.boxes.cls):
                        break

        print(ld_width)

        image_old = cv2.line(image_line.copy(), p2, max_fat_pt, (255, 255, 0), 10)
        image_old = cv2.line(image_old, p1, p2, (0, 0, 255), 20)

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
                img_final = correct_measurements(
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
                img_final = correct_measurements(
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
                img_final = correct_measurements(
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
        except:
            print("Error in creating measurements, likely bad mask or bad call for orientation of fat relative to muscle")
        
        cv2.imwrite(
            os.path.join("output/annotated_images", f"{result.path.split('/')[-1].split('.')[0]}_annotated.JPG"),
            img_final,
        )
    create_table(id_list, ld_depth_list, ld_width_list, muscle_to_fat_list, args)

if __name__ == "__main__":
    main()
