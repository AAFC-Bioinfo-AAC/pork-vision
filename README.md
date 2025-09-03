# PORK-VISION

[![FR](https://img.shields.io/badge/lang-FR-yellow.svg)](README_FR.md)
[![EN](https://img.shields.io/badge/lang-EN-blue.svg)](README.md)

![License](https://img.shields.io/badge/License-GPLv3-blue.svg)

<!-- omit in toc -->
## Table of Contents

- [PORK-VISION](#pork-vision)
  - [About](#about)
  - [Credits](#credits)
  - [Contribution](#contribution)
  - [License](#license)
  - [References](#references)
  - [Citation](#citation)

---

## About

Exports from the Canadian pork industry generate $5 billion per year. Primal cuts with desirable quality attributes, especially loins, bellies and butts, are sold at premium prices in international markets, such as Japan. Current methods used for measuring pork quality, both in-line and under research conditions, are conducted through mainly subjective methods and manual testing on the loin primal. Fully automated systems are not usually available for the collection of quality data in pork primals or pork chops, and adoption of the few available technologies able to evaluate some quality traits has been limited due to high costs and operational requirements.

Here we developed a Python-based image analysis pipeline using computer vision and deep learning techniques to automate the evaluation of center pork chops of loin primals (gold standard location for evaluation of pork quality) based on the most important quality attributes required by domestic and international buyers. Using an existing large pork phenomics image bank and dataset generated at the AAFC Lacombe Research and Development Centre (Lacombe, AB), the system was developed and validated under conditions mimicking commercial processing. It replicates manual workflows traditionally performed using ImageJ and custom macros, streamlining the process while maintaining compatibility with the Canadian pork colour and marbling standards.

The pipeline extracts quantitative measurements such as muscle width and depth, fat depth, marbling percentage, and color score from standardized pork chop images. It is designed to process large batches efficiently, making it well-suited for research and industry applications alike. Developed entirely in Python, the system leverages libraries such as PyTorch, OpenCV, and NumPy, and integrates:

- Deep Learning Models:
  - A segmentation model to isolate fat and muscle regions
  - A YOLOv11 object detection model for identifying embedded color standards
- Custom Algorithms:
  - Image preprocessing and measurement algorithms for geometry and intensity-based analysis
  
For technical details, installation and usage, see [`User guide`](./docs/user-guide.md).

---

## Credits

Developed by a multidisciplinary team of bioinformaticians and meat science experts at Lacombe Research and Development centre, Agriculture & Agri-Food Canada. For individual contributions, see the [CREDITS](CREDITS.md) file.

---

## Contribution

If you would like to contribute to this project, please review the guidelines in [CONTRIBUTING.md](CONTRIBUTING.md) and ensure you adhere to our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## License

This project is distributed under the GPLv3 License. For complete details and copyright information, see the [LICENSE](LICENSE) file.

---

## References

References to tools and software used here can be found in the [CITATIONS.md](CITATIONS.md) file.

---

## Citation

If you use this project in your work, please cite it using the [CITATION.cff](CITATION.cff) file.
