# SPDX-License-Identifier: GPL-3.0-or-later
# © His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food Canada, 2025.
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

def compute_percentage(mask, roi):
    if mask is None or roi is None:
        return float("nan")

    roi_pixels = roi > 0             
    total = roi_pixels.sum()
    if total == 0:
        return float("nan")

    marbling = np.logical_and(mask == 0, roi_pixels).sum()
    return marbling * 100.0 / total

def save_marbling_csv(id_list, fat_percentage_list, output_csv_path):
    df = pd.DataFrame({
        'image_id':       id_list,
        'fat_percentage': fat_percentage_list
    })
    df.to_csv(output_csv_path, index=False)