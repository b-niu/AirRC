# AirRC

Code accompanying the paper:

> Liu, J., Zhang, Z., Niu, B. et al. A Custom Annotated Dataset for Segmentation of
> Pulmonary Veins, Arteries, and Airways. *Sci Data* **12**, 1806 (2025).
> [https://doi.org/10.1038/s41597-025-06074-6](https://doi.org/10.1038/s41597-025-06074-6)

These files are excerpts of the pipeline described in that paper, provided so that the
methods can be read alongside their implementation. They cover the pre-segmentation
algorithms, the loss functions and the two-stage training strategy. Data loading, the
stratified splits, the augmentation transforms, the optimizer configuration and the
sliding-window inference live outside this repository and are omitted.

## `pulmonary_structure_segmentation_tools.py` — coarse pre-segmentation

These routines are deliberately crude. The masks they produce are far from usable as they
are and serve only as a starting point for extensive manual correction; they reduce the
annotation effort rather than replace it. The refinement that follows is what determines the
final anatomical accuracy.

*Airways.* A seed point is placed in the trachea and grown with SimpleITK's
`ConfidenceConnectedImageFilter`. The volume is first clamped to `[-1000, 0]` HU so that
lumen and lung parenchyma become comparable, and the growing is controlled by a sensitivity
`multiplier` (3.0). A 3D opening followed by a closing with a 1-voxel radius removes
speckle and smooths the border. A second, threshold-based variant is included as an
alternative: `ConnectedThresholdImageFilter` on the clamped volume, followed by connected
component volume filtering that discards leaks into the parenchyma.

*Vessels.* Processed slice by slice. Each slice is clamped to `[-2000, 2000]` HU,
normalized to 0-255, and thresholded with Otsu to obtain a first lung field. That field is
refined with a secondary threshold equal to the mean intensity of the pixels above the Otsu
threshold times a lung refinement factor (0.75). The mask is then cleaned by clearing border
pixels, median filtering, dropping border-connected components, filling holes with a flood
fill, and an erosion/dilation pair for smoothing. On the contrast-enhanced slice (factor
1.5), the mean intensity of the non-lung pixels times a vessel adjustment factor (1.25)
gives the adaptive threshold for vessel candidates, which are finally restricted to the lung
mask. The 2D results are stacked back into a 3D volume.

## `config_loss.py` — loss functions

* `TimiLoss`, the specialized lumen loss. It combines a soft Dice and cross-entropy term
  with a focal union term that penalizes false negatives at the periphery of the airway
  tree, with the two contributions weighted 0.5 and 1.0 (α and β in the paper). The
  formulation follows Team timi's winning solution to the ATM'22 challenge
  ([reference implementation](https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/tree/main/ATM22-Challenge-Top5-Solution/team_timi));
  only the interface and the configuration are kept here, and `forward` is a placeholder.
* `DeepSupervisionLossBase` applies the supervision scheme of both stages: the outputs are
  combined as a weighted sum with exponentially decaying, normalized weights.
* `PerClassLoss` implements the class-weighted objective of Stage 2. Each foreground class
  is turned into an independent binary problem, the airway lumen is scored with `TimiLoss`
  and the remaining classes with a MONAI `DiceCELoss`, and the per-class results are
  combined through `class_weights` — raising one entry concentrates the training on that
  class. The paper uses 1.0 for the lumen and 0.5 for the others.
* `DeepSupervisionTimiLoss`, `DeepSupervisionDiceCELoss` and `DeepSupervisionPerClassLoss`
  wrap the above in the deep-supervision scheme, so that the two stages only differ in the
  base loss they pass in. These wrappers were written by hand for the paper; MONAI now
  provides an official implementation of the same idea,
  `monai.losses.DeepSupervisionLoss`, which we recommend using instead.

## `train_stage1.py` — Stage 1, baseline model

A baseline model trained to establish a strong initial segmentation of every foreground
structure. The backbone is a MONAI `DynUNet` with residual blocks and four deep supervision
heads, and the objective is the weighted Deep Supervision Dice and Cross-Entropy loss
described in the paper. Inputs are 1 mm isotropic volumes with the intensity clipping and
z-score normalization used in the paper's preprocessing, and validation follows the
stratified five-fold split. The reported metric is the mean Dice over the foreground classes.

## `train_stage2.py` — Stage 2, refinement model

A second model, trained for 150 epochs with two changes with respect to the baseline.

*Objective and optimizer.* `DeepSupervisionPerClassLoss`, i.e. the class-weighted loss
described above applied across the deep supervision outputs, with the same SGD optimizer and
polynomial learning-rate schedule as in Stage 1.

*Hard case mining.* This is what actually distinguishes the refinement stage. Every case
carries two additional maps next to image and label — a `sample_weight` and a `loss_weight`,
assembled by `add_weight_paths` — and the transforms run in `hard_case` mode, applying the
same random spatial deformation to all four arrays. Training therefore concentrates on the
regions the baseline model got wrong, and the corresponding errors are given more weight in
the loss. The weights of the baseline are loaded from the Stage 1 checkpoint before training.

## Environment

MONAI 1.5.0 and PyTorch Lightning 2.5.2 for the training scripts, on Python 3.11. The
pre-segmentation script was developed with SimpleITK 2.3.1, OpenCV 4.10.0 and
scikit-image 0.24.0.

## Data

The AirRC annotations (254 cases, 1 mm isotropic NIfTI masks and `metadata.xlsx`) are
available on Figshare under CC BY 4.0
([https://doi.org/10.6084/m9.figshare.26878867](https://doi.org/10.6084/m9.figshare.26878867)).
The source CT scans are not redistributed here; obtain them from LUNA16 and resample them to
1 mm isotropic spacing before use.

## License

GPL-3.0.
