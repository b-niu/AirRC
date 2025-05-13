# AirRC

# Coarse Pulmonary Structure Segmentation Tools for CT Images

This repository provides Python scripts for performing initial coarse segmentation of pulmonary airways and blood vessels from computed tomography (CT) images. These tools implement the automated pre-segmentation pipeline described in the [Link to your Data Descriptor paper/preprint, if available, otherwise mention it's associated with a manuscript] for generating the AirRC dataset.

The primary goal of these scripts is to produce rough initial segmentations that can significantly expedite the manual annotation process by providing a starting point for expert radiologists. The final accuracy of segmentations for datasets like AirRC relies on subsequent meticulous manual refinement.

## Features

*   **Airway Segmentation:**
    *   Seed-based 3D region growing.
    *   Two methods provided:
        1.  `ConfidenceConnectedImageFilter` from SimpleITK.
        2.  `ConnectedThresholdImageFilter` from SimpleITK, applied on HU-clamped images, followed by connected component volume filtering to remove leakage into lung parenchyma.
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

*   Python 3.8+
*   Required Python packages:
    *   SimpleITK
    *   NumPy
    *   OpenCV-Python (`opencv-python`)
    *   Scikit-image (`scikit-image`)

You can install the required packages using pip:
```bash
pip install SimpleITK numpy opencv-python scikit-image
