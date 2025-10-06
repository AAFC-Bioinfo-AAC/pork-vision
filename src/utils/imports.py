
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (c) 2025 His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food, 2025.
# Pork-vision: pork chop image analysis pipeline.
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

import os
import cv2
import numpy as np
import pandas as pd
import argparse


def parse_args():
    '''
    Standard argument parser
    model_path: Path to the fat segmentation model
    color_model_path: Path to the Canadian Standard detection model
    image_path: Path to the image folder (data)
    output_path: Path to output.
    '''
    parser = argparse.ArgumentParser(description="Run PorkVision Inference and Analysis")
    parser.add_argument("--model_path", type=str, default="src/models/Yolo_MuscleFat_Segment_98epoch.pt")
    parser.add_argument("--color_model_path", type=str, default="src/models/color_100_last.pt")
    parser.add_argument("--image_path", type=str, default="data/")
    parser.add_argument("--output_path", type=str, default="output/")
    parser.add_argument("--minimal", action="store_true",   help="Run in minimal mode (no extra files)")
    parser.add_argument("--debug",   action="store_true",   help="Write per‐image debug logs")
    parser.add_argument("--outputs", type=str, default="all",
        help=(
            "Comma-separated list of analyses to run e.g. python src/main.py --outputs measurement,marbling"
            "Use 'all' to run everything."
        )
    )
    args = parser.parse_args()

    outs = [o.strip().lower() for o in args.outputs.split(",")]
    if "all" in outs:
        outs = ["measurement", "marbling", "colour"]
    args.outputs = outs

    return args