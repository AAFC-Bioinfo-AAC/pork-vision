
# CHANGELOG

## v1.0.0 — Initial Public Release

**Release Date:** 2025-06-xx

---

### Overview

Porkvision `v1.0.0` marks the first full, production-ready release of an end-to-end image analysis pipeline for pork chop evaluation, tailored to Canadian industry grading standards. Developed with scalability and automation in mind, this version integrates deep learning–based segmentation, classical computer vision, and robust post-processing to deliver quantitative meat quality assessments at scale.

---

### Key Features

- **Full Modular Pipeline**
  - Preprocessing, orientation standardization, measurement, marbling, and coloring all modularized.

- **Deep Learning Models**
  - Fat/muscle segmentation via YOLOv11 & SAM
  - Color standard detection via custom-trained YOLO model

- **Measurement Tools**
  - Muscle width, depth, and fat thickness calculation with mm/px conversion from ruler detection

- **Marbling Analysis**
  - Intramuscular fat quantification using contrast enhancement and geometric filtering

- **Color Scoring**
  - Color score classification based on proximity to Canadian pork color reference palettes

- **Output System**
  - Clean export of annotated images, ROIs, and CSV summaries (`marbling.csv`, `measurement.csv`, `colouring.csv`)

- **Batch Processing Ready**
  - Designed to scale across hundreds of images using SLURM or local execution

---

### Setup Improvements

- Conda environment and dependency pinning provided
- SLURM-compatible shell script for HPC batch runs
- Debug mode and optional module execution flags

---

### Example Outputs
- `output/annotated_images/`: measurement overlays  
- `output/marbling/`: marbling and fat masks  
- `output/rois/`: region-of-interest files for manual QA  
- `output/colouring/`: color palette detections and LUT comparisons

---

### Acknowledgements

Developed by a multidisciplinary team of bioinformaticians and meat science experts at Agriculture & Agri-Food Canada.

For more details, see the [README](README.md) or cite using the [CITATION.cff](CITATION.cff) file.
