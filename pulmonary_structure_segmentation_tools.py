# -*- coding: utf-8 -*-
"""
pulmonary_structure_segmentation_tools.py

This script provides functions for coarse segmentation of pulmonary airways
and blood vessels from CT images.
- Airway segmentation uses seed-based region growing (ConfidenceConnected
  or ConnectedThreshold with volume filtering).
- Vessel segmentation uses a 2D slice-wise approach with adaptive
  thresholding and morphological operations.

These methods are intended to produce initial, coarse segmentations
suitable for subsequent expert manual refinement.
"""
import os
import math
import numpy as np
import SimpleITK as sitk
import cv2
from skimage.measure import label as skimage_label # Renamed to avoid conflict

# --- Helper Functions ---

def read_image(file_path):
    """
    Reads NIfTI, MHD, or other image formats supported by SimpleITK.
    Args:
        file_path (str): Path to the image file.
    Returns:
        SimpleITK.Image: The loaded image, or None if file not found.
    Raises:
        FileNotFoundError: If the input file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")
    try:
        image = sitk.ReadImage(file_path)
        print(f"Read image: {file_path}, Size: {image.GetSize()}, Spacing: {image.GetSpacing()}")
        return image
    except RuntimeError as e:
        print(f"Error reading image {file_path}: {e}")
        return None

def write_image(image, file_path):
    """
    Writes a SimpleITK image to the specified file path.
    Creates output directory if it doesn't exist.
    Args:
        image (SimpleITK.Image): The image to write.
        file_path (str): The path to save the image.
    """
    output_dir = os.path.dirname(file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    try:
        sitk.WriteImage(image, file_path)
        print(f"Saved image to: {file_path}")
    except RuntimeError as e:
        print(f"Error writing image {file_path}: {e}")

def fill_hole_cv(image_slice_np):
    """
    Fills holes in a binary 2D numpy array using OpenCV's flood fill.
    Assumes input is a single slice (2D).
    Args:
        image_slice_np (np.ndarray): Input binary 2D image slice.
    Returns:
        np.ndarray: Image slice with holes filled.
    """
    flood_filled_image = image_slice_np.copy()
    height, width = image_slice_np.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood_filled_image, mask, (0, 0), 255) # Flood from corner
    inverted_flood_filled_image = cv2.bitwise_not(flood_filled_image)
    filled_image = image_slice_np | inverted_flood_filled_image
    return filled_image

def round_up_to_odd(num):
    """
    Rounds a number up to the nearest odd integer.
    Args:
        num (float or int): The number to round.
    Returns:
        int: The smallest odd integer greater than or equal to num.
    """
    num = int(np.ceil(num)) # Ensure it's at least this integer
    if num % 2 == 0:
        num += 1
    return num

# --- Airway Segmentation Functions ---

def _validate_seed_point(image_size_xyz, seed_point_voxel_coords_zyx):
    """Internal helper to validate seed point."""
    seed_z, seed_y, seed_x = seed_point_voxel_coords_zyx
    if not (0 <= seed_x < image_size_xyz[0] and \
            0 <= seed_y < image_size_xyz[1] and \
            0 <= seed_z < image_size_xyz[2]):
        print(f"Error: Seed point {seed_point_voxel_coords_zyx} (z,y,x) is outside image bounds {image_size_xyz[::-1]} (z,y,x).")
        return None
    return (int(seed_x), int(seed_y), int(seed_z)) # SITK order (x,y,z)

def segment_airways_coarse_confidence_connected(ct_image_sitk, seed_point_voxel_coords_zyx,
                                                multiplier=3.0, # Manuscript default
                                                iterations=2,
                                                initial_radius=2, # Manuscript seems to use 1 for morph_radius
                                                morph_kernel_radius_xyz=(1, 1, 1)):
    """
    Performs coarse airway segmentation using SimpleITK's ConfidenceConnectedImageFilter.

    Args:
        ct_image_sitk (SimpleITK.Image): Input CT image.
        seed_point_voxel_coords_zyx (tuple): Seed point (z, y, x) voxel indices.
        multiplier (float): Controls sensitivity (std dev multiplier). Higher values
                            allow more variation. Manuscript default is 3.0.
        iterations (int): Number of times to recalculate statistics for region growing.
        initial_radius (int): Radius for initial statistics calculation around the seed.
        morph_kernel_radius_xyz (tuple): Radius in voxels (x,y,z) for final 3D
                                         morphological opening and closing.
                                         Set to (0,0,0) to disable. Manuscript default (1,1,1).

    Returns:
        SimpleITK.Image: Binary airway mask (UInt8), or an empty mask on error/no segmentation.
    """
    print(f"Starting airway segmentation (ConfidenceConnected) with multiplier={multiplier}...")
    image_size_xyz = ct_image_sitk.GetSize() # (x, y, z)

    seed_point_sitk_order_xyz = _validate_seed_point(image_size_xyz, seed_point_voxel_coords_zyx)
    if seed_point_sitk_order_xyz is None:
        return sitk.Image(image_size_xyz, sitk.sitkUInt8) # Return empty mask
    print(f"Validated seed point (x,y,z): {seed_point_sitk_order_xyz}")

    # 1. Preprocessing: Clamp HU values [-1000, 0] as per manuscript
    ct_array = sitk.GetArrayFromImage(ct_image_sitk)
    airway_hu_clamped_array = np.clip(ct_array, -1000, 0)
    airway_image_clamped_sitk = sitk.GetImageFromArray(airway_hu_clamped_array.astype(np.int16))
    airway_image_clamped_sitk.CopyInformation(ct_image_sitk)

    # 2. Region Growing (Confidence Connected)
    conf_connected_filter = sitk.ConfidenceConnectedImageFilter()
    conf_connected_filter.SetSeedList([seed_point_sitk_order_xyz])
    conf_connected_filter.SetNumberOfIterations(iterations)
    conf_connected_filter.SetMultiplier(multiplier)
    conf_connected_filter.SetInitialNeighborhoodRadius(initial_radius)
    conf_connected_filter.SetReplaceValue(1) # Output pixel value for segmented region

    try:
        airway_region_grown = conf_connected_filter.Execute(airway_image_clamped_sitk)
        num_pixels = np.sum(sitk.GetArrayViewFromImage(airway_region_grown))
        print(f"ConfidenceConnected region growing completed. Segmented {num_pixels} voxels.")
        if num_pixels == 0:
             print("Warning: ConfidenceConnected region growing resulted in an empty mask. Try adjusting multiplier or seed point.")
             empty_mask = sitk.Image(image_size_xyz, sitk.sitkUInt8)
             empty_mask.CopyInformation(ct_image_sitk)
             return empty_mask
    except RuntimeError as e:
        print(f"Error during ConfidenceConnected region growing: {e}")
        empty_mask = sitk.Image(image_size_xyz, sitk.sitkUInt8)
        empty_mask.CopyInformation(ct_image_sitk)
        return empty_mask

    # 3. Optional Morphological Refinement (as per manuscript: opening and closing)
    airway_final = airway_region_grown
    if morph_kernel_radius_xyz != (0, 0, 0) and all(r > 0 for r in morph_kernel_radius_xyz) :
        print(f"Applying 3D morphological opening and closing with radius {morph_kernel_radius_xyz}...")
        try:
            # Manuscript: opening and closing operations (kernel radius: 1 voxel)
            # Order can matter. Let's try opening then closing.
            opened_mask = sitk.BinaryMorphologicalOpening(airway_region_grown, morph_kernel_radius_xyz)
            airway_final = sitk.BinaryMorphologicalClosing(opened_mask, morph_kernel_radius_xyz)
        except RuntimeError as e:
            print(f"Warning: Morphological operation failed: {e}. Using mask directly from region growing.")
            # airway_final remains airway_region_grown
    else:
        print("Skipping morphological operations or radius is zero.")

    print("Airway segmentation (ConfidenceConnected method) finished.")
    airway_final = sitk.Cast(airway_final, sitk.sitkUInt8)
    return airway_final


def segment_airways_coarse_connected_threshold_vol_filtered(
    ct_image_sitk, seed_point_voxel_coords_zyx,
    lower_threshold_on_clamped, upper_threshold_on_clamped,
    max_volume_threshold_voxels,
    morph_kernel_radius_xyz=(1, 1, 1)
):
    """
    Performs coarse airway segmentation using ConnectedThresholdImageFilter
    applied ON THE HU-CLAMPED IMAGE, followed by volume filtering to remove
    large connected components (likely lung parenchyma).

    Args:
        ct_image_sitk (SimpleITK.Image): Input CT image.
        seed_point_voxel_coords_zyx (tuple): Seed point (z, y, x) voxel indices.
        lower_threshold_on_clamped (float): Lower HU bound for growing (applied to values in [-1000, 0]).
        upper_threshold_on_clamped (float): Upper HU bound for growing (applied to values in [-1000, 0]).
        max_volume_threshold_voxels (int): Maximum number of voxels allowed for a
                                           connected component to be kept. Components
                                           larger than this are removed.
        morph_kernel_radius_xyz (tuple): Radius for final morphological opening/closing.

    Returns:
        SimpleITK.Image: Binary airway mask (UInt8), or an empty mask on error/no segmentation.
    """
    print(f"Starting airway segmentation (ConnectedThreshold on Clamped HU) with range [{lower_threshold_on_clamped}, {upper_threshold_on_clamped}] and max volume {max_volume_threshold_voxels} voxels...")
    image_size_xyz = ct_image_sitk.GetSize()

    seed_point_sitk_order_xyz = _validate_seed_point(image_size_xyz, seed_point_voxel_coords_zyx)
    if seed_point_sitk_order_xyz is None:
        return sitk.Image(image_size_xyz, sitk.sitkUInt8)
    print(f"Validated seed point (x,y,z): {seed_point_sitk_order_xyz}")

    # 1. Preprocessing: Clamp HU values [-1000, 0]
    ct_array = sitk.GetArrayFromImage(ct_image_sitk)
    airway_hu_clamped_array = np.clip(ct_array, -1000, 0)
    airway_image_clamped_sitk = sitk.GetImageFromArray(airway_hu_clamped_array.astype(np.int16))
    airway_image_clamped_sitk.CopyInformation(ct_image_sitk)

    seed_value_in_clamped = airway_image_clamped_sitk.GetPixel(seed_point_sitk_order_xyz)
    print(f"Value at seed point in clamped [-1000, 0] image: {seed_value_in_clamped}")
    if not (lower_threshold_on_clamped <= seed_value_in_clamped <= upper_threshold_on_clamped):
        print(f"Error: Seed point value ({seed_value_in_clamped}) is OUTSIDE the specified threshold range [{lower_threshold_on_clamped}, {upper_threshold_on_clamped}]. Growth cannot start.")
        return sitk.Image(image_size_xyz, sitk.sitkUInt8)

    # 2. Region Growing (Connected Threshold on Clamped Image)
    connected_filter = sitk.ConnectedThresholdImageFilter()
    connected_filter.SetSeedList([seed_point_sitk_order_xyz])
    connected_filter.SetLower(float(lower_threshold_on_clamped))
    connected_filter.SetUpper(float(upper_threshold_on_clamped))
    connected_filter.SetReplaceValue(1)

    try:
        airway_region_grown = connected_filter.Execute(airway_image_clamped_sitk)
        num_pixels_initial = np.sum(sitk.GetArrayViewFromImage(airway_region_grown))
        print(f"Initial ConnectedThreshold region growing completed. Segmented {num_pixels_initial} voxels.")
        if num_pixels_initial == 0:
            print("Warning: Initial region growing resulted in an empty mask.")
            return sitk.Image(image_size_xyz, sitk.sitkUInt8)
    except RuntimeError as e:
        print(f"Error during ConnectedThreshold region growing: {e}")
        return sitk.Image(image_size_xyz, sitk.sitkUInt8)

    # 3. Volume Filtering
    print(f"Applying volume filtering (max allowed volume = {max_volume_threshold_voxels} voxels)...")
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True) # Consider if 3D full connectivity is desired
    labeled_components = cc_filter.Execute(airway_region_grown)

    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.Execute(labeled_components)

    num_labels = shape_stats.GetNumberOfLabels()
    print(f"Found {num_labels} connected components.")

    # Create a new image to store the filtered result
    # Initialize with airway_region_grown to keep components that meet criteria
    # Then selectively remove components that are too large.
    # Alternative: start with zeros and add small components.
    # For safety, let's ensure the seed component is considered,
    # though if it's part of a huge leak, this won't save it.
    final_mask_after_volume_filter = sitk.Image(image_size_xyz, sitk.sitkUInt8)
    final_mask_after_volume_filter.CopyInformation(ct_image_sitk)
    output_array = sitk.GetArrayFromImage(final_mask_after_volume_filter) # ZYX
    labeled_array = sitk.GetArrayFromImage(labeled_components) # ZYX

    kept_component_count = 0
    for i in range(1, num_labels + 1): # Labels are 1-based
        num_voxels_in_component = shape_stats.GetNumberOfPixels(i)
        if num_voxels_in_component < max_volume_threshold_voxels:
            output_array[labeled_array == i] = 1
            kept_component_count +=1

    final_mask_after_volume_filter = sitk.GetImageFromArray(output_array)
    final_mask_after_volume_filter.CopyInformation(ct_image_sitk)

    print(f"Kept {kept_component_count} components after volume filtering.")
    if kept_component_count == 0:
        print("Warning: Volume filtering removed all components. Check max_volume_threshold or initial segmentation.")
        return sitk.Image(image_size_xyz, sitk.sitkUInt8)

    # 4. Optional Final Morphological Refinement
    airway_final = final_mask_after_volume_filter
    if morph_kernel_radius_xyz != (0, 0, 0) and all(r > 0 for r in morph_kernel_radius_xyz):
        print(f"Applying final 3D morphological opening and closing with radius {morph_kernel_radius_xyz}...")
        try:
            opened_mask = sitk.BinaryMorphologicalOpening(final_mask_after_volume_filter, morph_kernel_radius_xyz)
            airway_final = sitk.BinaryMorphologicalClosing(opened_mask, morph_kernel_radius_xyz)
        except RuntimeError as e:
            print(f"Warning: Final morphological operation failed: {e}. Using volume-filtered mask.")
            # airway_final remains final_mask_after_volume_filter
    else:
        print("Skipping final morphological operations or radius is zero.")

    print("Airway segmentation (ConnectedThreshold + Volume Filter) finished.")
    airway_final = sitk.Cast(airway_final, sitk.sitkUInt8)
    return airway_final

# --- Vessel Segmentation Function ---

def segment_pulmonary_vessels_coarse(ct_image_sitk,
                                     lung_refinement_factor=0.75, # Manuscript default
                                     vessel_adjustment_factor=1.25, # Manuscript default
                                     output_dir="output_debug",
                                     save_intermediate=False):
    """
    Performs coarse pulmonary vessel segmentation using a 2D slice-wise approach.
    This function implements the automated pipeline described in the manuscript
    for initial coarse vessel segmentation.

    Args:
        ct_image_sitk (SimpleITK.Image): Input CT image.
        lung_refinement_factor (float): Factor to refine Otsu's threshold for lung mask.
                                        Manuscript default 0.75.
        vessel_adjustment_factor (float): Factor to adjust threshold for vessel candidates.
                                          Manuscript default 1.25.
        output_dir (str): Directory to save intermediate images if save_intermediate is True.
        save_intermediate (bool): If True, saves intermediate images for debugging.

    Returns:
        SimpleITK.Image: Binary vessel mask (UInt8).
    """
    print("Starting coarse pulmonary vessel segmentation...")
    if save_intermediate and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ct_array_zyx = sitk.GetArrayFromImage(ct_image_sitk) # Z, Y, X
    image_size_xyz = ct_image_sitk.GetSize()
    spacing_xyz = ct_image_sitk.GetSpacing()

    vessel_mask_array_zyx = np.zeros_like(ct_array_zyx, dtype=np.uint8)

    for i in range(ct_array_zyx.shape[0]): # Iterate through Z slices
        print(f"Processing slice {i+1}/{ct_array_zyx.shape[0]} for vessel segmentation...")
        slice_yx = ct_array_zyx[i, :, :].astype(np.float64) # Work with float for calculations

        # --- 1. Preprocessing (as per manuscript) ---
        # Clamp HU values to [-2000, 2000]
        slice_clamped_yx = np.clip(slice_yx, -2000, 2000)
        # Normalize to 0-255 grayscale range
        min_val, max_val = np.min(slice_clamped_yx), np.max(slice_clamped_yx)
        if max_val == min_val: # Avoid division by zero for blank slices
            slice_normalized_yx = np.zeros_like(slice_clamped_yx, dtype=np.uint8)
        else:
            slice_normalized_yx = 255 * (slice_clamped_yx - min_val) / (max_val - min_val)
        slice_normalized_yx = slice_normalized_yx.astype(np.uint8)

        if save_intermediate: write_image(sitk.GetImageFromArray(slice_normalized_yx), os.path.join(output_dir, f"slice_{i}_01_normalized.png"))

        # --- 2. Initial Lung Field Approximation (Otsu's) ---
        otsu_thresh_val, lung_mask_otsu_yx = cv2.threshold(slice_normalized_yx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if save_intermediate: write_image(sitk.GetImageFromArray(lung_mask_otsu_yx), os.path.join(output_dir, f"slice_{i}_02_otsu_lung.png"))

        # --- 3. Refine Lung Mask ---
        # "secondary threshold, calculated by multiplying the mean intensity of pixels
        # above the Otsu threshold by a lung refinement factor (default: 0.75)"
        pixels_above_otsu = slice_normalized_yx[lung_mask_otsu_yx > 0]
        if len(pixels_above_otsu) > 0:
            mean_intensity_above_otsu = np.mean(pixels_above_otsu)
            secondary_thresh_val = lung_refinement_factor * mean_intensity_above_otsu
            _, lung_mask_refined_yx = cv2.threshold(slice_normalized_yx, secondary_thresh_val, 255, cv2.THRESH_BINARY)
        else: # If Otsu mask is empty, refined mask is also empty
            lung_mask_refined_yx = np.zeros_like(slice_normalized_yx, dtype=np.uint8)

        if save_intermediate: write_image(sitk.GetImageFromArray(lung_mask_refined_yx), os.path.join(output_dir, f"slice_{i}_03_refined_lung_thresh.png"))

        # --- Further Refinements (as per manuscript) ---
        # "clearing border pixels"
        lung_mask_refined_yx[0, :] = 0; lung_mask_refined_yx[-1, :] = 0
        lung_mask_refined_yx[:, 0] = 0; lung_mask_refined_yx[:, -1] = 0

        # "applying adaptive median filtering" - OpenCV medianBlur is not adaptive in size by default
        # Using a fixed reasonable size, or manuscript might imply custom adaptive logic.
        # Let's use a fixed median blur as a proxy for now.
        # Adaptively sized kernels for morph ops later use round_up_to_odd.
        median_blur_kernel_size = round_up_to_odd(7 / 512 * image_size_xyz[0]) # Using X dimension for kernel scaling
        lung_mask_refined_yx = cv2.medianBlur(lung_mask_refined_yx, median_blur_kernel_size)
        if save_intermediate: write_image(sitk.GetImageFromArray(lung_mask_refined_yx), os.path.join(output_dir, f"slice_{i}_04_median_blurred.png"))


        # "removing border-connected components"
        # This means if a component touches the border, remove it.
        # A common way is to flood fill from borders on inverted mask.
        # Here, skimage_label is used to identify components.
        # Components connected to [0,0] etc. are background.
        labels_yx, num_labels = skimage_label(lung_mask_refined_yx, background=0, return_num=True, connectivity=2) # skimage.measure.label
        border_labels = set()
        border_labels.update(labels_yx[0, :])    # Top row
        border_labels.update(labels_yx[-1, :])   # Bottom row
        border_labels.update(labels_yx[:, 0])    # Left col
        border_labels.update(labels_yx[:, -1])   # Right col
        for l_val in border_labels:
            if l_val != 0: # Don't remove background label
                lung_mask_refined_yx[labels_yx == l_val] = 0
        if save_intermediate: write_image(sitk.GetImageFromArray(lung_mask_refined_yx), os.path.join(output_dir, f"slice_{i}_05_border_cleared.png"))


        # "filling holes via a flood-fill algorithm"
        lung_mask_refined_yx = fill_hole_cv(lung_mask_refined_yx)
        if save_intermediate: write_image(sitk.GetImageFromArray(lung_mask_refined_yx), os.path.join(output_dir, f"slice_{i}_06_holes_filled.png"))

        # "smoothing using adaptively sized morphological erosion and dilation operations"
        # Manuscript doesn't specify kernel shape or exact adaptive sizing logic here.
        # Using a small, fixed kernel for general smoothing effect.
        # Kernel sizes scaled by image dimension (e.g., X)
        # Assuming a small radius like 1-2 pixels effectively at 1mm spacing
        # These values might need to be factors of spacing if images are not ~1mm iso
        erosion_kernel_size = round_up_to_odd(max(1, 1.0 * image_size_xyz[0] / 512.0)) # e.g. 1 if 512, scaled
        dilation_kernel_size = round_up_to_odd(max(1, 1.0 * image_size_xyz[0] / 512.0))
        kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_kernel_size, erosion_kernel_size))
        kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))

        lung_mask_refined_yx = cv2.erode(lung_mask_refined_yx, kernel_e)
        lung_mask_refined_yx = cv2.dilate(lung_mask_refined_yx, kernel_d)
        if save_intermediate: write_image(sitk.GetImageFromArray(lung_mask_refined_yx), os.path.join(output_dir, f"slice_{i}_07_morph_smoothed.png"))


        # --- 4. Identify Vessel Candidates ---
        # "linear contrast enhancement (factor: 1.5) was applied to the normalized slice"
        # Using OpenCV's convertScaleAbs for simplicity, alpha is contrast factor
        slice_contrast_enhanced_yx = cv2.convertScaleAbs(slice_normalized_yx, alpha=1.5, beta=0)
        if save_intermediate: write_image(sitk.GetImageFromArray(slice_contrast_enhanced_yx), os.path.join(output_dir, f"slice_{i}_08_contrast_enhanced.png"))


        # "After temporarily masking out the lung regions, the mean intensity of the
        # remaining non-lung pixels was computed."
        non_lung_pixels_yx = slice_contrast_enhanced_yx[lung_mask_refined_yx == 0] # Use refined lung mask
        if len(non_lung_pixels_yx) > 0:
            mean_intensity_non_lung = np.mean(non_lung_pixels_yx)
        else: # If lung mask covers everything, or image is blank
            mean_intensity_non_lung = 0 # Avoid errors, vessel mask will be empty

        # "A final adaptive vessel threshold was derived by multiplying this mean intensity
        # by a vessel adjustment factor (default: 1.25)."
        vessel_thresh_val = vessel_adjustment_factor * mean_intensity_non_lung

        # "Applying this threshold to the contrast-enhanced image yielded binary masks
        # of potential vessel structures"
        # We only want vessels *within* the lung mask.
        _, slice_vessel_mask_yx = cv2.threshold(slice_contrast_enhanced_yx, vessel_thresh_val, 255, cv2.THRESH_BINARY)

        # Mask out regions outside the lung
        slice_vessel_mask_yx[lung_mask_refined_yx == 0] = 0
        if save_intermediate: write_image(sitk.GetImageFromArray(slice_vessel_mask_yx), os.path.join(output_dir, f"slice_{i}_09_vessel_mask.png"))

        vessel_mask_array_zyx[i, :, :] = slice_vessel_mask_yx

    # Combine slices into 3D volume
    vessel_mask_sitk = sitk.GetImageFromArray(vessel_mask_array_zyx)
    vessel_mask_sitk.CopyInformation(ct_image_sitk) # Ensure spacing, origin, direction are copied

    print("Coarse pulmonary vessel segmentation finished.")
    return vessel_mask_sitk


# --- Example Usage ---
if __name__ == "__main__":
    # --- Common Paths ---
    # !!! REPLACE WITH YOUR ACTUAL PATHS !!!
    # Example CT scan path (NIfTI or MHD)
    example_ct_path = r"path/to/your/ct_scan.nii.gz"
    # General output directory for all segmentation results
    general_output_dir = r"path/to/your/output_segmentations"
    os.makedirs(general_output_dir, exist_ok=True)

    # --- Load Image ---
    ct_image = None
    if os.path.exists(example_ct_path):
        ct_image = read_image(example_ct_path)
    else:
        print(f"Error: Example CT file not found at '{example_ct_path}'. Please update the path.")
        exit()

    if ct_image is None:
        print("Failed to load CT image. Exiting.")
        exit()

    # --- Airway Segmentation Example ---
    print("\n--- Example: Airway Segmentation ---")
    # Define a seed point in (Z, Y, X) voxel coordinates for airway
    # This needs to be carefully chosen for each scan, typically in the trachea.
    # You might need to inspect your image in a viewer to find good coordinates.
    # Example: (slice_index, row_index, column_index)
    # Adjust based on your image dimensions (ct_image.GetSize() is X,Y,Z)
    image_size_xyz_main = ct_image.GetSize()
    # A generic seed near the top-center, assuming trachea is there.
    # THIS IS A GUESS AND WILL LIKELY NEED ADJUSTMENT PER SCAN.
    example_airway_seed_zyx = (
        int(image_size_xyz_main[2] * 0.85), # ~85% down from top in Z (SITK Z is last)
        int(image_size_xyz_main[1] * 0.5),  # Y midpoint
        int(image_size_xyz_main[0] * 0.5)   # X midpoint
    )
    print(f"Using example airway seed (Z,Y,X): {example_airway_seed_zyx} for image size (X,Y,Z): {image_size_xyz_main}")


    # Method 1: Confidence Connected for Airways
    output_airway_cc_path = os.path.join(general_output_dir, "airway_mask_confidence_connected.nii.gz")
    airway_mask_cc = segment_airways_coarse_confidence_connected(
        ct_image_sitk=ct_image,
        seed_point_voxel_coords_zyx=example_airway_seed_zyx,
        multiplier=3.0, # From manuscript
        morph_kernel_radius_xyz=(1,1,1) # From manuscript
    )
    if airway_mask_cc and np.sum(sitk.GetArrayViewFromImage(airway_mask_cc)) > 0:
        write_image(airway_mask_cc, output_airway_cc_path)
    else:
        print("ConfidenceConnected airway segmentation failed or produced an empty mask.")

    # Method 2: Connected Threshold with Volume Filtering for Airways
    output_airway_ctf_path = os.path.join(general_output_dir, "airway_mask_connected_threshold_vol_filtered.nii.gz")
    # These thresholds are applied on HU values clamped to [-1000, 0]
    airway_lower_thresh_clamped = -1000 # Typical for air
    airway_upper_thresh_clamped = -750  # Tune this: higher values (e.g. -700) are more restrictive
    # Max volume for airway components (in voxels). Highly dependent on image resolution
    # and desired extent of airway tree. Start large, then reduce if too much lung leaks.
    # If airways are cut short, increase this. This value is a wild guess.
    airway_max_volume_voxels = 500000 # Needs significant tuning per dataset/resolution

    airway_mask_ctf = segment_airways_coarse_connected_threshold_vol_filtered(
        ct_image_sitk=ct_image,
        seed_point_voxel_coords_zyx=example_airway_seed_zyx,
        lower_threshold_on_clamped=airway_lower_thresh_clamped,
        upper_threshold_on_clamped=airway_upper_thresh_clamped,
        max_volume_threshold_voxels=airway_max_volume_voxels,
        morph_kernel_radius_xyz=(1,1,1) # From manuscript
    )
    if airway_mask_ctf and np.sum(sitk.GetArrayViewFromImage(airway_mask_ctf)) > 0 :
        write_image(airway_mask_ctf, output_airway_ctf_path)
    else:
        print("ConnectedThreshold+VolumeFilter airway segmentation failed or produced an empty mask.")


    # --- Vessel Segmentation Example ---
    print("\n--- Example: Vessel Segmentation ---")
    output_vessel_path = os.path.join(general_output_dir, "vessel_mask_coarse.nii.gz")
    # output_dir for intermediate vessel slices (optional)
    vessel_intermediate_dir = os.path.join(general_output_dir, "vessel_intermediate_slices")

    vessel_mask = segment_pulmonary_vessels_coarse(
        ct_image_sitk=ct_image,
        lung_refinement_factor=0.75, # From manuscript
        vessel_adjustment_factor=1.25, # From manuscript
        output_dir=vessel_intermediate_dir,
        save_intermediate=True # Set to False to disable saving intermediate slices
    )
    if vessel_mask and np.sum(sitk.GetArrayViewFromImage(vessel_mask)) > 0:
        write_image(vessel_mask, output_vessel_path)
    else:
        print("Vessel segmentation failed or produced an empty mask.")

    print("\nAll example segmentations complete.")
