
# PorkVision Documentation Index

---

## Description

The following documentation provides deatiled guidance on navigating the PorkVision repository. It is structured to guide users of all experience levels through setup, usage, and customization of the codebase. Each section of the repository's documentation provides detailed instructions and references for key functionalities.

---

## Table of Contents

- [Description](#description)
- [Technical Resources](#technical-resources)
  - [Configuration Files](#configuration-files)
  - [Tool Documentation](#tool-documentation)
  - [Environment Setup](#environment-setup)
  - [Testing and Validation](#testing-and-validation)
- [Glossary](#glossary)
  - [Automated Muscle and Fat Measurements](#automated-muscle-and-fat-measurements)
  - [Image Processing and Segmentation](#image-processing-and-segmentation)
  - [Geometry and Measurements](#geometry-and-measurements)
  - [Software and Tools](#software-and-tools)
- [Additional Documentation Files](#additional-documentation-files)
  - [README.md](#readmemd)
  - [Parameters.md](#parametersmd)
  - [ABCC_RCBA_Guide](#abcc_rcba_guide)

---

## Technical Resources

### Configuration Files
The project uses configuration files for environment setup and dependencies. These include:
- `environment.yml`: Defines the required Python libraries and versions.
- `LICENSE`: Specifies the terms and conditions for using this project.

### Tool Documentation
Below are links to the key tools and libraries leveraged in PorkVision:
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [NumPy](https://numpy.org/doc/stable/)
- [OpenCV](https://docs.opencv.org/4.x/index.html)
- [pandas](https://pandas.pydata.org/docs/)
- [SciPy](https://scipy.org/)
- [Shapely](https://shapely.readthedocs.io/en/stable/)

### Environment Setup
The recommended Python version is 3.9. To set up the environment:
1. Use `conda` to create the environment using the `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate the environment:
   ```bash
   conda activate yolosam_env
   ```
3. Install additional dependencies if required:
   ```bash
   pip install lsq-ellipse
   ```

### Testing and Validation
To ensure the repository is functioning as expected:
1. Use the provided sample images in the `raw_images/` directory.
2. Run the pipeline and validate the outputs in the `runs/` directory.

---

## Glossary

### Automated Muscle and Fat Measurements

- **LD Muscle Width:** The longest horizontal line segment that spans across the LD muscle region. Used as a measure of pork carcass muscle size.

- **LD Muscle Depth:** A vertical line segment measured 7 cm from the midline of the carcass, extending perpendicular to the skin. This metric is part of the Canadian grading system.

- **Upper Fat Depth:** The portion of the vertical segment that extends through the fat tissue, measured above the muscle. Calculated by extending the muscle depth line until it intersects with the upper fat boundary.

### Image Processing and Segmentation

- **Image Segmentation:** A computer vision technique that partitions an image into regions of interest. In this project, it is used to isolate the LD muscle region.

- **Binary Mask:** A two-dimensional array where pixels belonging to the region of interest are marked as `1` (foreground) and all others as `0` (background).

- **U-Net Model:** A CNN architecture designed for semantic segmentation tasks. It was used in this project to segment the LD muscle.

- **Bounding Box:** A rectangle enclosing the region of interest in an image. Its dimensions can provide key measurements such as width and height.

- **Angle of Inclination:** The angle by which an object (e.g., pork loin) deviates from horizontal alignment. Calculated via least-squares ellipse fitting and corrected to standardize measurements.

### Geometry and Measurements

- **Bounding Box Midpoints:** The geometric midpoints of bounding box edges, used to calculate the orientation or alignment of the LD muscle.

- **Least-Squares Ellipse Fitting:** A mathematical method to approximate an ellipse around a set of points, used to estimate the angle of inclination and align the muscle region.

- **Contour Points:** A sequence of points outlining the boundary of a region of interest, such as the LD muscle mask.

### Software and Tools

- **YOLOv8:** A deep learning-based object detection model.

- **Segment Anything:** A model developed for zero-shot image segmentation tasks. Used for isolating regions like the LD muscle in images.

- **OpenCV:** A computer vision library used for image processing tasks, such as rotation, overlaying measurements, and contour detection.

---

## Additional Documentation Files

### Project Introduction
Refer to [This video](https://001gc.sharepoint.com/:v:/r/sites/42732/mmg1/BioinfoConf_Recordings_and_Resources/BioinfoConf2024_recordings/BioinfoConf2024_2.3_EdwardYakubovitch.mp4?csf=1&web=1&e=k5jFn0) for an introduction to the project.

### README.md
Refer to [README.md](/README.md) for an overview of the repository.

### Parameters.md
Refer to [parameters.md](/docs/parameters.md) for clarification on function inputs throughout the source code.

### ABCC_RCBA_Guide
Refer to the [ABCC_RCBA_Guide](https://github.com/AAFC-Bioinformatics/ABCC_RCBA_Guide) for additional context and supplementary materials that align with this and other AAFC projects.

---

For any questions or contributions, please refer to the [CONTRIBUTING.md](.github/CONTRIBUTING.md) file.