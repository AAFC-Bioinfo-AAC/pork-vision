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