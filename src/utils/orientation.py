# SPDX-License-Identifier: GPL-3.0-or-later
# © His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food, 2025.
# Pork-vision: pork chop image analysis pipeline.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License,
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

def initial_orientation_correction(original_image, muscle_mask, fat_mask, debug_messages, rotation=cv2.ROTATE_90_CLOCKWISE):
    '''
    Corrects the initial orientation (if the image is upside down or sideways relative to the fat up).
    Input: Original_image, muscle_mask, fat_mask.
    Output: Rotated_image, muscle_mask, fat_mask
    '''
    height, width, _ = original_image.shape
    if width<height: # If the image is vertical.
        debug_messages.append("Image is vertical, correcting")
        debug_messages.append(f"Height before rotation = {height}, width before rotation = {width}")
        rotated_image = cv2.rotate(original_image, rotation)
        rotated_muscle_mask = cv2.rotate(muscle_mask, rotation)
        rotated_fat_mask = cv2.rotate(fat_mask, rotation)
        height, width, _ = rotated_image.shape
        debug_messages.append(f"Height after rotation {height}, width after rotation {width}")
    else:
        rotated_image = original_image
        rotated_muscle_mask = muscle_mask
        rotated_fat_mask = fat_mask
    fat_pixels = np.where(rotated_fat_mask == 255)
    muscle_pixels = np.where(rotated_muscle_mask == 255)


    # Get the maximum y value for the muscle mask
    muscley_value = np.max(muscle_pixels[0])

    # Get the x values where y = muscley_value
    musclex_values = muscle_pixels[1][muscle_pixels[0] == muscley_value]

    # If multiple x values exist at this y, choose the first one:
    musclex_value = musclex_values[0]

    # Get the y values where x = musclex_value in fat mask
    faty_values_at_musclex = fat_pixels[0][fat_pixels[1] == musclex_value]

    # Get the maximum y value at this x position in the fat mask
    faty_value = np.max(faty_values_at_musclex)
    #If the fat is below the muscle rotate to fix.
    if faty_value > muscley_value:
        debug_messages.append(f"CORRECTING: Bottom fat detected at {faty_value}, while the bottom muscle is {muscley_value}, so fat is below muscle.")
        if width>height:
            rotated_image = cv2.rotate(rotated_image, cv2.ROTATE_180)
            rotated_muscle_mask = cv2.rotate(rotated_muscle_mask, cv2.ROTATE_180)
            rotated_fat_mask = cv2.rotate(rotated_fat_mask, cv2.ROTATE_180)
            return rotated_image,rotated_muscle_mask,rotated_fat_mask,debug_messages

    debug_messages.append(f"GOOD: Fat detected at {faty_value} while muscle is at {muscley_value} so fat above muscle")
    return rotated_image, rotated_muscle_mask, rotated_fat_mask, debug_messages
    
def orient_muscle_and_fat_using_adjacency(original_image, muscle_mask, fat_mask, outlier, debug_messages):
    """
    Orients the image so that the thin strip of fat adjacent to the muscle is on top.

    Steps:
    1. Get bounding boxes for muscle and the adjacent fat region.
    2. Compare bounding box centers to decide 0°, +90°, -90°, or 180° rotation.

    Parameters:
    - original_image (np.ndarray): Original image.
    - muscle_mask (np.ndarray): Binary mask of the muscle region.
    - fat_mask (np.ndarray): Binary mask of the fat region.

    Returns:
    - rotated_image (np.ndarray): Rotated image.
    - rotated_muscle_mask (np.ndarray): Rotated muscle mask.
    - rotated_fat_mask (np.ndarray): Rotated fat mask.
    - final_angle (int): The rotation angle in degrees (0, 90, -90, or 180).
    """

    # 1. Find bounding box of muscle
    muscle_contours, _ = cv2.findContours(muscle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not muscle_contours:
        outlier = "Y"
        debug_messages.append("No muscle found. Skipping orientation.")
        return original_image, muscle_mask, fat_mask, 0, outlier, debug_messages
    rotated_image, rotated_muscle_mask, rotated_fat_mask, debug_messages = initial_orientation_correction(original_image, muscle_mask, fat_mask, debug_messages)
    height,width, _ = rotated_image.shape
    if width<height:
        outlier = "Y"
        debug_messages.append("ERROR: Image is vertical after rotation.")
    return rotated_image, rotated_muscle_mask, rotated_fat_mask, 0, outlier, debug_messages