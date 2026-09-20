# AirRC

# Coarse Pulmonary Structure Segmentation Tools for CT Images

This repository provides Python scripts for performing initial coarse segmentation of pulmonary airways and blood vessels from computed tomography (CT) images. These tools implement the automated pre-segmentation pipeline described in the data descriptor paper [Liu, J., Zhang, Z., Niu, B. et al. A Custom Annotated Dataset for Segmentation of Pulmonary Veins, Arteries, and Airways. *Sci Data* **12**, 1806 (2025)](https://doi.org/10.1038/s41597-025-06074-6) for generating the AirRC (Airway and Pulmonary Vessel Structural Representation in CT) dataset.

The primary goal of these scripts is to produce rough initial segmentations that can significantly expedite the manual annotation process by providing a starting point for expert radiologists. The final accuracy of segmentations for datasets like AirRC relies on subsequent meticulous manual refinement.

## Features

*   **Airway Segmentation:**
    *   Seed-based 3D region growing.
    *   Two methods provided:
        1.  `ConfidenceConnectedImageFilter` from SimpleITK. **This is the method used in the manuscript's pre-segmentation pipeline.**
        2.  `ConnectedThresholdImageFilter` from SimpleITK, applied on HU-clamped images, followed by connected component volume filtering to remove leakage into lung parenchyma. **This is an additional alternative offered by this repository and is not part of the published pipeline.**
    *   Preprocessing includes HU value clamping specific for airway visualization.
    *   Optional morphological refinement (opening and closing).
*   **Pulmonary Vessel Segmentation:**
    *   2D slice-wise processing.
    *   Lung field approximation using Otsu's thresholding followed by refinement steps (border clearing, hole filling, morphological smoothing).
    *   Adaptive thresholding for vessel candidate identification based on non-lung pixel intensity, after contrast enhancement.
    *   Outputs a 3D binary mask of potential vessel structures.

## Script Overview

*   `pulmonary_structure_segmentation_tools.py`: Contains all core functions for reading images, performing airway segmentation, performing vessel segmentation, and writing output masks. Includes an example usage block (`if __name__ == "__main__":`) for demonstration.

## Prerequisites

*   Python 3.8+ (the manuscript pipeline was developed with Python 3.11).
*   Required Python packages (versions used in the manuscript in parentheses):
    *   SimpleITK (2.3.1)
    *   NumPy
    *   OpenCV-Python (`opencv-python`, 4.10.0)
    *   Scikit-image (`scikit-image`, 0.24.0)

> **Input requirements:** the pipeline was developed and validated on CT volumes resampled to **1 mm x 1 mm x 1 mm** isotropic voxels. Please resample your scans to this spacing before running these scripts; otherwise the voxel-based kernel sizes (which are scaled relative to a 512-pixel in-plane field of view) will not behave as intended.

You can install the required packages using pip:
```bash
pip install SimpleITK numpy opencv-python scikit-image
```

## Citation

If you use these tools or the AirRC dataset, please cite the corresponding data descriptor:

> Liu, J., Zhang, Z., Niu, B. et al. A Custom Annotated Dataset for Segmentation of Pulmonary Veins, Arteries, and Airways. *Sci Data* **12**, 1806 (2025). https://doi.org/10.1038/s41597-025-06074-6

```bibtex
@article{liu2025airrc,
  title   = {A Custom Annotated Dataset for Segmentation of Pulmonary Veins, Arteries, and Airways},
  author  = {Liu, Jian and Zhang, Zheng and Niu, Bing and Kang, Shuai and Ren, Juan and Wang, Lei and Xu, Kai},
  journal = {Scientific Data},
  volume  = {12},
  pages   = {1806},
  year    = {2025},
  doi     = {10.1038/s41597-025-06074-6}
}
```

## Data Availability

*   **AirRC annotations** (254 cases, 1 mm isotropic NIfTI masks plus `metadata.xlsx`): Figshare, https://doi.org/10.6084/m9.figshare.26878867 (CC BY 4.0).
*   **Source CT images** are not redistributed here. Obtain the LUNA16 scans separately (Zenodo, https://doi.org/10.5281/zenodo.3723295 and https://doi.org/10.5281/zenodo.4121926, CC BY 4.0) and resample them to 1 mm isotropic spacing before use.
