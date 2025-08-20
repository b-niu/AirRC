from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.utils import one_hot

from monai.losses import DiceCELoss


class TimiLoss(nn.Module):
    """
    TimiLoss implementation for airway segmentation.

    This loss function combines DiceCE loss with a specialized focal union loss
    for enhanced airway segmentation performance. It's designed specifically for
    the ATM-22 challenge winning solution.

    For complete implementation details, please refer to:
    https://github.com/EndoluminalSurgicalVision-IMR/ATM-22-Related-Work/tree/main/ATM22-Challenge-Top5-Solution/team_timi
    """

    def __init__(
        self,
        num_classes: int,
        airway_class_index: int = 1,
        num_spatial_dims: int = 3,
        dice_ce_weight: float = 0.5,
        focal_union_weight: float = 1.0,
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
    ):
        super().__init__()
        # Implementation details removed for simplicity
        # Please refer to the GitHub repository for full implementation
        pass

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        patch_weight: torch.Tensor,
    ) -> torch.Tensor:
        # Implementation details removed for simplicity
        # Please refer to the GitHub repository for full implementation
        # This is a placeholder that returns a dummy loss value
        return torch.tensor(0.0, device=logits.device)


class DeepSupervisionLossBase(nn.Module):
    def __init__(
        self,
        deep_supr_num: int,
        weights: Optional[np.ndarray] = None,
        **kwargs_base_loss,
    ):
        super().__init__()
        self.num_levels = deep_supr_num + 1

        # Create weights
        if weights is not None:
            self.weights = torch.from_numpy(weights.astype(np.float32))
        else:
            weights_array = np.array(
                [1 / (2**i) for i in range(self.num_levels)], dtype=np.float32
            )
            weights_array[-1] = 0.0
            total_weight = weights_array.sum()
            if total_weight > 0:
                weights_array = weights_array / total_weight
            self.weights = torch.from_numpy(weights_array)

    def forward(
        self, outputs: torch.Tensor, labels: torch.Tensor, **kwargs_loss: Any
    ) -> torch.Tensor:
        # If not deep supervision format, calculate single loss
        if outputs.ndim != 6:
            return self.base_loss(outputs, labels, **kwargs_loss)

        # Calculate weighted sum of losses across all levels
        total_loss = torch.tensor(0.0, device=outputs.device)
        for level_index in range(self.num_levels):
            level_weight = self.weights[level_index]
            if level_weight < 1e-8:
                continue
            level_output = outputs[:, level_index, ...]
            level_loss = self.base_loss(level_output, labels, **kwargs_loss)
            total_loss += level_weight * level_loss

        return total_loss


class DeepSupervisionTimiLoss(DeepSupervisionLossBase):
    def __init__(self, deep_supr_num: int, **kwargs):
        super().__init__(deep_supr_num, **kwargs)
        self.base_loss = TimiLoss(**kwargs)


class PerClassLoss(nn.Module):
    def __init__(
        self,
        class_weights: Dict[int, float],
        lumen_class_index: int,
        num_spatial_dims: int = 3,
        dice_ce_weight: float = 0.5,
        focal_union_weight: float = 1.0,
    ):
        super().__init__()
        self.class_weights = class_weights
        self.lumen_idx = lumen_class_index

        # Create loss functions for each class
        self.lumen_loss_fn = TimiLoss(
            num_classes=2,
            airway_class_index=1,
            num_spatial_dims=num_spatial_dims,
            dice_ce_weight=dice_ce_weight,
            focal_union_weight=focal_union_weight,
        )

        self.other_class_loss_fn = DiceCELoss(
            num_classes=2,
            num_spatial_dims=num_spatial_dims,
            include_background=False,
            batch_dice=True,
        )

    def forward(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        patch_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=logits.device)

        for class_idx, weight in self.class_weights.items():
            if weight <= 0:
                continue

            # Create binary logits and target for this class
            binary_logits = torch.cat(
                [logits[:, 0:1, ...], logits[:, class_idx : class_idx + 1, ...]], dim=1
            )

            binary_target = (target == class_idx).long()

            # Calculate loss for this class
            if class_idx == self.lumen_idx:
                current_loss = self.lumen_loss_fn(
                    binary_logits, binary_target, patch_weight
                )
            else:
                current_loss = self.other_class_loss_fn(binary_logits, binary_target)

            total_loss += weight * current_loss

        return total_loss


class DeepSupervisionPerClassLoss(DeepSupervisionLossBase):
    def __init__(self, deep_supr_num: int, **kwargs):
        super().__init__(deep_supr_num, **kwargs)
        self.base_loss = PerClassLoss(**kwargs)
