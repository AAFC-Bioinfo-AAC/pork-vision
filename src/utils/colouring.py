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

from utils.imports import *
from scipy import stats
import numpy as np

def create_coloring_standards(image, model, image_id, output_dir, outlier, minimal, debug_messages):
    canadian_standard_unsorted = []
    result = model.predict(image, save=False)[0]
    detection_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # We use RGB for future sorting.
    if result == None:
        debug_messages.append("Create Coloring Standards: No result received from YOLO model")
        return None, 'Y', debug_messages
    debug_messages.append("Parsing through bounding boxes.")
    for box in result.boxes: # For each box determine if it's a duplicate of an already existing class, if not then append class_id + mode rgb.
        duplicate = False
        class_id = int(box.cls[0])
        for items in canadian_standard_unsorted:
            if items[0] == class_id:
                duplicate = True
        if duplicate == True: 
            debug_messages.append("Duplicate color standard detected. Skipping")
            continue
        mode_rgb = get_mode_rgb(detection_image, box)
        id_rgb = [class_id, mode_rgb] # Class ID 0 = Canadian standard 6, Class ID 6 = Canadian standard 0
        debug_messages.append(f"{id_rgb}")
        canadian_standard_unsorted.append(id_rgb)
    debug_messages.append(f"Unsorted standard chart: {canadian_standard_unsorted}")
    debug_messages.append("Starting insertion sort")
    canadian_standard_sorted = insertion_sort(canadian_standard_unsorted)
    canadian_array = np.vstack(canadian_standard_sorted).astype(np.float32) #Turns into a singular list.
    debug_messages.append(f"Canadian Array (Std 6 to Std 0): {canadian_array}")
    if minimal == False:
        debug_messages.append(f"Saving color detect image")
        os.makedirs(output_dir, exist_ok=True)
        base_output_dir = os.path.join(output_dir, image_id)
        os.makedirs(base_output_dir, exist_ok=True)
        save_path = f'{base_output_dir}/{image_id}_Color_Detect.jpg'
        result.save(save_path)
    return canadian_array, outlier, debug_messages


# RGB values for Canadian lean color standards made from 102,103,104,105,107,109,110 2024 images
class_to_std = {0: "Canadian_Std6",
                1 : "Canadian_Std5",
                2 : "Canadian_Std4",
                3 : "Canadian_Std3",
                4 : "Canadian_Std2", 
                5 : "Canadian_Std1", 
                6 : "Canadian_Std0",}

def get_mode_rgb(image, bbox):
    try:
        x_min, y_min, x_max, y_max = bbox.xyxy[0]  # Focus interest on the bounding box.
        x_min = int(x_min)
        y_min = int(y_min)
        x_max = int(x_max)
        y_max = int(y_max)
        focus = image[y_min:y_max, x_min:x_max].reshape(-1,3)  #Crop into the bounding box and reshape the image into 2D array consisting of rows of pixel containing RGB
        mode_rgb = stats.mode(focus, axis=0).mode
        return mode_rgb
    except:
        return (0,0,0)
    
def insertion_sort(canadian_standard_unsorted):
    for i in range(1, len(canadian_standard_unsorted)):
        key = canadian_standard_unsorted[i]
        #print(f"Current key is {key}")
        index = i - 1
        #print(f"Key - 1 (index) is {index}")

        while index >= 0 and key[0]<canadian_standard_unsorted[index][0]:
            canadian_standard_unsorted[index + 1] = canadian_standard_unsorted[index]
            index = index - 1
        canadian_standard_unsorted[index+1] = key
    #print()
    #print(f"Sorted array in Ascending order {canadian_standard_unsorted}")
    for sublist in canadian_standard_unsorted:
        del sublist[0]
    #print(canadian_standard_unsorted)
    return canadian_standard_unsorted