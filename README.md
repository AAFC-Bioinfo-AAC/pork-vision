# PORK-VISION

[![FR](https://img.shields.io/badge/lang-FR-yellow.svg)](README_FR.md)
[![EN](https://img.shields.io/badge/lang-EN-blue.svg)](README.md)

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

---

## Documentation

For technical details, including installation and usage instructions, please see the [**`User Guide`**](./docs/user-guide.md).

---

## Acknowledgements

This project was developed by a multidisciplinary team of bioinformaticians and meat science experts at the Lacombe Research and Development Centre, Agriculture & Agri-Food Canada.

- **Credits**: For a list of individual contributions, see [CREDITS](CREDITS.md).

- **Citation**: To cite this project, click the **`Cite this repository`** button on the right-hand sidebar.

- **Contributing**: To contribute to this project, review the guidelines provided in the [CONTRIBUTING](CONTRIBUTING.md) file and adhere to our [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md).

- **References**: For a list of key resources and software used here, see [REFERENCES](REFERENCES.md).

---

## License

This project is licensed under the **GNU General Public License v3.0 (GPLv3)**. See [LICENSE](LICENSE) for details.

**Copyright (c)** His Majesty the King in Right of Canada, as represented by the Minister of Agriculture and Agri-Food, 2025.
