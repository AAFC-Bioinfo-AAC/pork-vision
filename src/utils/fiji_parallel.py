#!/usr/bin/env python

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

"""
Parallel FIJI processing utilities for pork-vision pipeline.
Replaces sequential batch processing with parallel execution.
"""

import os
import sys
import subprocess
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time

FIJI_CMD = os.environ.get("FIJI_CMD", "fiji")


def process_single_image_marbling(
    image_filename: str,
    regions_dir: str,
    masks_dir: str,
    macro_path: str
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single image for marbling analysis using FIJI.
    
    Args:
        image_filename: Name of the crop file (e.g., "sample_crop.png")
        regions_dir: Path to regions directory
        masks_dir: Path to masks output directory
        macro_path: Path to FIJI macro file
        
    Returns:
        Tuple of (filename, success, error_message)
    """
    try:
        # Create a temporary macro that processes just this one image
        base_name = image_filename.replace("_crop.png", "")
        
        # Build single-image macro content
        # This replicates the logic from batch_marble.ijm for a single image
        single_macro = f'''
setBatchMode(true);

regDir  = "{regions_dir}" + File.separator;
maskDir = "{masks_dir}" + File.separator;

// Process single image: {image_filename}
open(regDir + "{image_filename}");
open(regDir + "{base_name}_roi.png");

run("Create Selection");
selectWindow("{image_filename}");
run("Restore Selection");
setBackgroundColor(0,0,0); 
run("Clear Outside");

close("*_roi.png");

run("Subtract Background...", "rolling=25");
run("8-bit");

AUTO_THRESHOLD = 5000;
getRawStatistics(pixcount);
limit = pixcount/10;
threshold = pixcount/AUTO_THRESHOLD;
nBins = 256;
getHistogram(values, histA, nBins);
h = -1; found = false;
do{{
    counts = histA[++h];
    if (counts > limit) counts = 0;
    found = counts > threshold;
}}while((!found)&&(h < histA.length-1));
hmin = values[h];

h = histA.length;
do{{
    counts = histA[--h];
    if (counts > limit) counts = 0;
    found = counts > threshold;
}}while((!found)&&(h > 0));
hmax = values[h];
setMinAndMax(hmin, hmax);
run("Apply LUT");

run("Auto Local Threshold", "method=Otsu radius=5 white");
run("Enlarge...", "enlarge=-15");
setBackgroundColor(0,0,0); 
run("Clear Outside");

run("Analyze Particles...", "size=14-Infinity show=[Masks] include");
saveAs("PNG", maskDir + "{image_filename}");

close("*");
'''
        
        # Write temporary macro file
        temp_macro_path = os.path.join(masks_dir, f"temp_{base_name}.ijm")
        with open(temp_macro_path, 'w') as f:
            f.write(single_macro)
        
        # Run FIJI on this single image
        cmd = [FIJI_CMD, "--headless", "--run", temp_macro_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout per image
        )
        
        # Clean up temp macro
        try:
            os.remove(temp_macro_path)
        except OSError:
            pass
        
        if result.returncode == 0:
            return (image_filename, True, None)
        else:
            return (image_filename, False, f"Return code {result.returncode}: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        return (image_filename, False, "FIJI process timed out")
    except Exception as e:
        return (image_filename, False, str(e))


def process_single_image_colour(
    image_filename: str,
    regions_dir: str,
    lean_dir: str,
    results_dir: str,
    macro_path: str
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single image for colour analysis using FIJI.
    
    Args:
        image_filename: Name of the crop file
        regions_dir: Path to regions directory
        lean_dir: Path to lean masks directory
        results_dir: Path to results output directory
        macro_path: Path to FIJI macro file
        
    Returns:
        Tuple of (filename, success, error_message)
    """
    try:
        base_name = image_filename.replace("_crop.png", "")
        
        # Build single-image colour macro
        # Note: You'll need to adapt this based on your actual batch_colour.ijm content
        # This is a template that you should modify based on your colour macro
        single_macro = f'''
setBatchMode(true);

regDir     = "{regions_dir}" + File.separator;
leanDir    = "{lean_dir}" + File.separator;
resultsDir = "{results_dir}" + File.separator;

// Process single image: {image_filename}
open(regDir + "{image_filename}");
open(leanDir + "{image_filename}");

// TODO: Replace this section with the actual logic from batch_colour.ijm
// This is a placeholder - you need to add your colour analysis steps here

// Save results
saveAs("Results", resultsDir + "{base_name}_colour.xls");
close("*");
'''
        
        # Write temporary macro
        temp_macro_path = os.path.join(results_dir, f"temp_{base_name}_colour.ijm")
        with open(temp_macro_path, 'w') as f:
            f.write(single_macro)
        
        # Run FIJI
        cmd = [FIJI_CMD, "--headless", "--run", temp_macro_path]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        # Clean up
        try:
            os.remove(temp_macro_path)
        except OSError:
            pass
        
        if result.returncode == 0:
            return (image_filename, True, None)
        else:
            return (image_filename, False, f"Return code {result.returncode}: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        return (image_filename, False, "FIJI process timed out")
    except Exception as e:
        return (image_filename, False, str(e))


def run_fiji_marbling_parallel(
    marbling_root: str,
    max_workers: Optional[int] = None
) -> Tuple[int, int]:
    """
    Run FIJI marbling analysis in parallel instead of batch mode.
    
    Args:
        marbling_root: Root directory for marbling processing
        max_workers: Number of parallel workers (default: CPU count // 2)
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    regions_dir = os.path.join(marbling_root, 'regions')
    masks_dir = os.path.join(marbling_root, 'masks')
    os.makedirs(masks_dir, exist_ok=True)
    
    macro_path = pathlib.Path(__file__).parent / "macros" / "batch_marble.ijm"
    
    # Find all crop images to process
    crop_files = [
        fn for fn in sorted(os.listdir(regions_dir))
        if fn.endswith("_crop.png")
    ]
    
    if not crop_files:
        print("No crop files found to process")
        return 0, 0
    
    print(f"Processing {len(crop_files)} images for marbling in parallel...")
    
    if max_workers is None:
        max_workers = max(1, os.cpu_count() // 2)
    
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_image_marbling,
                fn,
                regions_dir,
                masks_dir,
                str(macro_path)
            ): fn
            for fn in crop_files
        }
        
        # Process results as they complete
        for i, future in enumerate(as_completed(futures), 1):
            filename = futures[future]
            try:
                fname, success, error = future.result()
                if success:
                    successful += 1
                    print(f"[{i}/{len(crop_files)}] ✓ {fname}")
                else:
                    failed += 1
                    print(f"[{i}/{len(crop_files)}] ✗ {fname}: {error}")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(crop_files)}] ✗ {filename}: Unexpected error: {e}")
    
    elapsed = time.time() - start_time
    print(f"FIJI marbling parallel processing complete: {successful} successful, {failed} failed in {elapsed:.1f}s")
    
    return successful, failed


def run_fiji_colour_parallel(
    colour_root: str,
    lean_dir: str,
    max_workers: Optional[int] = None
) -> Tuple[int, int]:
    """
    Run FIJI colour analysis in parallel instead of batch mode.
    
    Args:
        colour_root: Root directory for colour processing
        lean_dir: Path to lean masks directory
        max_workers: Number of parallel workers
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    regions_dir = os.path.join(colour_root, 'regions')
    results_dir = os.path.join(colour_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    macro_path = pathlib.Path(__file__).parent / "macros" / "batch_colour.ijm"
    
    # Find all crop images
    crop_files = [
        fn for fn in sorted(os.listdir(regions_dir))
        if fn.endswith("_crop.png")
    ]
    
    if not crop_files:
        print("No crop files found for colour processing")
        return 0, 0
    
    print(f"Processing {len(crop_files)} images for colour in parallel...")
    
    if max_workers is None:
        max_workers = max(1, os.cpu_count() // 2)
    
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image_colour,
                fn,
                regions_dir,
                lean_dir,
                results_dir,
                str(macro_path)
            ): fn
            for fn in crop_files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            filename = futures[future]
            try:
                fname, success, error = future.result()
                if success:
                    successful += 1
                    print(f"[{i}/{len(crop_files)}] ✓ {fname}")
                else:
                    failed += 1
                    print(f"[{i}/{len(crop_files)}] ✗ {fname}: {error}")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(crop_files)}] ✗ {filename}: Unexpected error: {e}")
    
    elapsed = time.time() - start_time
    print(f"FIJI colour parallel processing complete: {successful} successful, {failed} failed in {elapsed:.1f}s")
    
    return successful, failed