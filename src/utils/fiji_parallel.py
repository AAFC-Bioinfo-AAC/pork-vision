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

Memory-optimized version for HPC/large batches.
"""

import os
import sys
import subprocess
import pathlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional
import time

FIJI_CMD = os.environ.get("FIJI_CMD", "fiji")


def process_single_image_marbling(
    image_filename: str,
    regions_dir: str,
    masks_dir: str,
    macro_path: str,
    quiet: bool = True
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single image for marbling analysis using FIJI.
    
    Args:
        image_filename: Name of the crop file (e.g., "sample_crop.png")
        regions_dir: Path to regions directory
        masks_dir: Path to masks output directory
        macro_path: Path to FIJI macro file
        quiet: Suppress FIJI warnings/output
        
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
        
        # Run FIJI on this single image with memory limit
        cmd = [FIJI_CMD, "--headless", "--mem", "2g", "--run", temp_macro_path]
        
        # Set environment variable to suppress warnings
        env = os.environ.copy()
        if quiet:
            env['SCIJAVA_LOG_LEVEL'] = 'ERROR'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
            env=env
        )
        
        # Clean up temp macro
        try:
            os.remove(temp_macro_path)
        except OSError:
            pass
        
        if result.returncode == 0:
            return (image_filename, True, None)
        else:
            # Check if it was killed (OOM)
            if result.returncode == 137 or 'Killed' in result.stderr:
                return (image_filename, False, "Process killed (likely out of memory)")
            return (image_filename, False, f"Return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        return (image_filename, False, "FIJI process timed out")
    except Exception as e:
        return (image_filename, False, str(e))


def process_single_image_colour(
    image_filename: str,
    regions_dir: str,
    lean_dir: str,
    results_dir: str,
    macro_path: str,
    quiet: bool = True
) -> Tuple[str, bool, Optional[str]]:
    """
    Process a single image for colour analysis using FIJI.
    
    Args:
        image_filename: Name of the colour file (e.g., "sample_colour.png")
        regions_dir: Path to regions directory
        lean_dir: Path to lean masks directory
        results_dir: Path to results output directory
        macro_path: Path to FIJI macro file
        quiet: Suppress FIJI warnings/output
        
    Returns:
        Tuple of (filename, success, error_message)
    """
    try:
        # Check if this is a colour file
        if not image_filename.endswith("_colour.png"):
            return (image_filename, False, "Not a colour file (must end with _colour.png)")
        
        base_name = image_filename.replace("_colour.png", "")
        
        # Check if required files exist
        std_path = os.path.join(regions_dir, f"{base_name}_std.txt")
        if not os.path.exists(std_path):
            return (image_filename, False, f"Missing standards file: {std_path}")
        
        # Build single-image colour macro with actual batch_colour.ijm logic
        single_macro = f'''
setBatchMode(true);
SHRINK_PX = 10;

regDir  = "{regions_dir}" + File.separator;
leanDir = "{lean_dir}" + File.separator;
resDir  = "{results_dir}" + File.separator;

base = "{base_name}";

C_R = newArray(7); C_G = newArray(7); C_B = newArray(7);
mid = newArray(6);
vHist = newArray(256); cHist = newArray(256); histA = newArray(256);

stdPath  = regDir + base + "_std.txt";
if (!File.exists(stdPath)) {{
    print("Skipping " + base + ": missing standards file");
    exit();
}}

stdTxt   = File.openAsString(stdPath);
stdLines = split(stdTxt, "\\r?\\n");

while (stdLines.length > 0 && trim(stdLines[stdLines.length - 1]) == "") {{
    stdLines = Array.slice(stdLines, 0, stdLines.length - 1);
}}

if (stdLines.length < 7) {{
    print("Skipping " + base + ": standards file has fewer than 7 lines");
    exit();
}}

for (s = 0; s < 7; s++) {{
    vals = split(trim(stdLines[s]), ",");
    if (vals.length < 3) {{
        print("Skipping " + base + ": bad standard line " + s);
        exit();
    }}
    C_R[s] = parseInt(vals[0]);
    C_G[s] = parseInt(vals[1]);
    C_B[s] = parseInt(vals[2]);
}}

if (C_G[0] < C_G[6]) {{
    R2=newArray(7); G2=newArray(7); B2=newArray(7);
    for (s=0; s<7; s++) {{ idx=6-s; R2[s]=C_R[idx]; G2[s]=C_G[idx]; B2[s]=C_B[idx]; }}
    for (s=0; s<7; s++) {{ C_R[s]=R2[s]; C_G[s]=G2[s]; C_B[s]=B2[s]; }}
}}

open(regDir + base + "_roi.png"); rename("roi");
run("8-bit"); setOption("BlackBackground", true); run("Convert to Mask");
for (t = 0; t < SHRINK_PX; t++) run("Erode");

open(leanDir + base + "_crop.png"); rename("marb");
run("8-bit"); setOption("BlackBackground", true); run("Convert to Mask");
run("Invert");

run("Image Calculator...", "image1=roi operation=AND image2=marb create");
rename("lean_mask");
close("roi"); close("marb");

open(regDir + base + "_G.png"); rename("Gchan"); run("8-bit");
if (!isOpen("Gchan")) {{
    print("Skipping " + base + ": could not open G plane");
    close("*");
    exit();
}}

selectWindow("Gchan"); run("Duplicate...", "title=Gviz");
selectWindow("Gviz");
run("Subtract Background...", "rolling=25");
AUTO_THRESHOLD = 5000; getRawStatistics(pixcount);
limit = pixcount/10; threshold = pixcount/AUTO_THRESHOLD;
getHistogram(vHist, histA, 256);
h=-1; do{{cnt=histA[++h]; if(cnt>limit) cnt=0;}}while(cnt<=threshold && h<histA.length-1);
hmin=vHist[h];
h=histA.length; do{{cnt=histA[--h]; if(cnt>limit) cnt=0;}}while(cnt<=threshold && h>0);
hmax=vHist[h]; setMinAndMax(hmin, hmax); run("Apply LUT");

for (s=0; s<6; s++) mid[s] = floor((C_G[s] + C_G[s+1]) / 2);
b0=mid[5]; b1=mid[4]; b2=mid[3]; b3=mid[2]; b4=mid[1]; b5=mid[0];

selectWindow("Gchan"); run("Duplicate...", "title=LABEL");
changeValues(0,      b0, 7);
changeValues(b0+1,   b1, 6);
changeValues(b1+1,   b2, 5);
changeValues(b2+1,   b3, 4);
changeValues(b3+1,   b4, 3);
changeValues(b4+1,   b5, 2);
changeValues(b5+1,   255, 1);
run("Image Calculator...", "image1=LABEL operation=AND image2=lean_mask create");
rename("LABEL_LEAN"); close("LABEL");

selectWindow("LABEL_LEAN"); getHistogram(vHist, cHist, 256);
leanPixels = 0; for (k=1; k<=7; k++) leanPixels += cHist[k];

run("Duplicate...", "title=C_R");
selectWindow("LABEL_LEAN"); run("Duplicate...", "title=C_G");
selectWindow("LABEL_LEAN"); run("Duplicate...", "title=C_B");
selectWindow("C_R"); for (s=0; s<7; s++) changeValues(s+1, s+1, C_R[s]);
selectWindow("C_G"); for (s=0; s<7; s++) changeValues(s+1, s+1, C_G[s]);
selectWindow("C_B"); for (s=0; s<7; s++) changeValues(s+1, s+1, C_B[s]);
selectWindow("C_G"); w=getWidth(); h=getHeight(); run("8-bit");
selectWindow("C_R"); run("8-bit"); if (getWidth()!=w || getHeight()!=h) run("Canvas Size...", "width="+w+" height="+h+" position=Top-Left");
selectWindow("C_B"); run("8-bit"); if (getWidth()!=w || getHeight()!=h) run("Canvas Size...", "width="+w+" height="+h+" position=Top-Left");
run("Merge Channels...", "red=C_R green=C_G blue=C_B create");
rename(base + "_Canadian_LUT"); saveAs("PNG", resDir + base + "_Canadian_LUT.png");

run("Clear Results");
for (s=0; s<7; s++) {{
    setResult("image_id",   s, base);
    setResult("Standard",   s, "CdnStd"+s);
    setResult("CdnCount",   s, cHist[s+1]);
    pct = 0.0; if (leanPixels > 0) pct = (cHist[s+1]*100.0)/leanPixels;
    setResult("CdnPercent", s, pct);
}}
updateResults();
saveAs("Results", resDir + base + "_colour.xls");

close("*");
'''
        
        # Write temporary macro
        temp_macro_path = os.path.join(results_dir, f"temp_{base_name}_colour.ijm")
        with open(temp_macro_path, 'w') as f:
            f.write(single_macro)
        
        # Run FIJI with memory limit
        cmd = [FIJI_CMD, "--headless", "--mem", "2g", "--run", temp_macro_path]
        
        # Set environment variable to suppress warnings
        env = os.environ.copy()
        if quiet:
            env['SCIJAVA_LOG_LEVEL'] = 'ERROR'
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            env=env
        )
        
        # Clean up
        try:
            os.remove(temp_macro_path)
        except OSError:
            pass
        
        if result.returncode == 0:
            return (image_filename, True, None)
        else:
            if result.returncode == 137 or 'Killed' in result.stderr:
                return (image_filename, False, "Process killed (likely out of memory)")
            return (image_filename, False, f"Return code {result.returncode}")
            
    except subprocess.TimeoutExpired:
        return (image_filename, False, "FIJI process timed out")
    except Exception as e:
        return (image_filename, False, str(e))


def run_fiji_marbling_parallel(
    marbling_root: str,
    max_workers: Optional[int] = None,
    quiet: bool = True
) -> Tuple[int, int]:
    """
    Run FIJI marbling analysis in parallel instead of batch mode.
    
    Memory-optimized for HPC/large batches.
    
    Args:
        marbling_root: Root directory for marbling processing
        max_workers: Number of parallel workers (default: min(4, CPU//2))
        quiet: Suppress FIJI warnings
        
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
    
    # Conservative worker count for HPC to avoid OOM
    if max_workers is None:
        max_workers = min(4, max(1, os.cpu_count() // 4))
    
    print(f"Using {max_workers} workers (conservative for memory)")
    
    successful = 0
    failed = 0
    failed_images = []
    
    start_time = time.time()
    
    # Use ThreadPoolExecutor instead of ProcessPoolExecutor
    # This reduces memory overhead significantly
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_single_image_marbling,
                fn,
                regions_dir,
                masks_dir,
                str(macro_path),
                quiet
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
                    if not quiet:
                        print(f"[{i}/{len(crop_files)}] ✓ {fname}")
                else:
                    failed += 1
                    failed_images.append((fname, error))
                    print(f"[{i}/{len(crop_files)}] ✗ {fname}: {error}")
            except Exception as e:
                failed += 1
                failed_images.append((filename, str(e)))
                print(f"[{i}/{len(crop_files)}] ✗ {filename}: Unexpected error: {e}")
    
    elapsed = time.time() - start_time
    print(f"FIJI marbling parallel processing complete: {successful} successful, {failed} failed in {elapsed:.1f}s")
    
    # If many failures, suggest adjustments
    if failed > len(crop_files) * 0.5:
        print(f"\n⚠️  High failure rate ({failed}/{len(crop_files)})")
        print("Suggestions:")
        print(f"  - Reduce max_workers (currently {max_workers})")
        print("  - Increase memory allocation in SLURM")
        print("  - Check failed images:")
        for fname, error in failed_images[:5]:  # Show first 5
            print(f"    {fname}: {error}")
    
    return successful, failed


def run_fiji_colour_parallel(
    colour_root: str,
    lean_dir: str,
    max_workers: Optional[int] = None,
    quiet: bool = True
) -> Tuple[int, int]:
    """
    Run FIJI colour analysis in parallel instead of batch mode.
    
    Memory-optimized for HPC/large batches.
    
    Args:
        colour_root: Root directory for colour processing
        lean_dir: Path to lean masks directory
        max_workers: Number of parallel workers
        quiet: Suppress FIJI warnings
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    regions_dir = os.path.join(colour_root, 'regions')
    results_dir = os.path.join(colour_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    macro_path = pathlib.Path(__file__).parent / "macros" / "batch_colour.ijm"
    
    # Find all colour images (not crop images!)
    colour_files = [
        fn for fn in sorted(os.listdir(regions_dir))
        if fn.endswith("_colour.png")
    ]
    
    if not colour_files:
        print("No colour files found for colour processing")
        return 0, 0
    
    print(f"Processing {len(colour_files)} images for colour in parallel...")
    
    # Conservative worker count
    if max_workers is None:
        max_workers = min(4, max(1, os.cpu_count() // 4))
    
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single_image_colour,
                fn,
                regions_dir,
                lean_dir,
                results_dir,
                str(macro_path),
                quiet
            ): fn
            for fn in colour_files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            filename = futures[future]
            try:
                fname, success, error = future.result()
                if success:
                    successful += 1
                    if not quiet:
                        print(f"[{i}/{len(colour_files)}] ✓ {fname}")
                else:
                    failed += 1
                    print(f"[{i}/{len(colour_files)}] ✗ {fname}: {error}")
            except Exception as e:
                failed += 1
                print(f"[{i}/{len(colour_files)}] ✗ {filename}: Unexpected error: {e}")
    
    elapsed = time.time() - start_time
    print(f"FIJI colour parallel processing complete: {successful} successful, {failed} failed in {elapsed:.1f}s")
    
    return successful, failed
