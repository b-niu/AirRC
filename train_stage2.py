import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import monai.losses
import numpy as np
import pytorch_lightning
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.data import DataLoader, PersistentDataset, decollate_batch, list_data_collate
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
from monai.networks.utils import one_hot
from monai.transforms import AsDiscrete, Compose, EnsureType
from monai.utils import set_determinism
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from torch.optim.lr_scheduler import _LRScheduler

sys.path.append(".")
import split
import transforms
import config_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def create_deep_supervision_layer_weights(
    deep_supervision_heads: int = 4,
) -> np.ndarray:
    num_outputs = deep_supervision_heads + 1
    deep_supervision_weights = np.array([(1 / (2**i)) for i in range(num_outputs)])
    deep_supervision_weights[-1] = 0
    normalized_weights = deep_supervision_weights / deep_supervision_weights.sum()
    return normalized_weights


class PolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        max_steps: int,
        exponent: float = 0.9,
        current_step: int = None,
    ):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        return self._last_lr


class Net(pytorch_lightning.LightningModule):
    def __init__(
        self,
        train_ds,
        val_ds,
        label_class_num=1,
        batch_size=4,
        num_workers=8,
        learning_rate=1e-3,
        pin_memory=True,
    ):
        super().__init__()
        num_classes = label_class_num + 1
        self.model = DynUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=num_classes,
            kernel_size=[3, 3, 3, 3, 3, 3],
            strides=[1, 2, 2, 2, 2, 2],
            upsample_kernel_size=[2, 2, 2, 2, 2],
            filters=[32, 64, 128, 256, 320, 320],
            norm_name=("INSTANCE_NVFUSER", {"eps": 1e-05, "affine": True}),
            act_name=("leakyrelu", {"negative_slope": 0.01, "inplace": True}),
            deep_supervision=True,
            deep_supr_num=4,
            res_block=True,
            trans_bias=True,
        )

        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv3d):
                if module.bias is None:
                    out_channels = module.out_channels
                    new_bias = torch.nn.Parameter(torch.zeros(out_channels))
                    module.bias = new_bias

        LUMEN_CLASS_INDEX = 1
        class_loss_weights = {
            1: 1.0,
            2: 0.5,
            3: 0.5,
            4: 0.5,
            5: 0.5,
        }

        self.loss_function = config_loss.DeepSupervisionPerClassLoss(
            deep_supr_num=4,
            class_weights=class_loss_weights,
            lumen_class_index=LUMEN_CLASS_INDEX,
            dice_ce_weight=0.5,
            focal_union_weight=1.0,
            num_spatial_dims=3,
        )
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.post_pred = Compose(
            [
                EnsureType("tensor"),
                AsDiscrete(argmax=True, to_onehot=num_classes),
            ]
        )
        self.post_label = Compose(
            [EnsureType("tensor"), AsDiscrete(to_onehot=num_classes)]
        )
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        patch_weight = batch["loss_weight"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels, patch_weight=patch_weight)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        patch_weight = batch["loss_weight"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels, patch_weight=patch_weight)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if outputs.ndim == 6:
            main_output = outputs[:, 0, ...]
        else:
            main_output = outputs

        outputs_post = [self.post_pred(i) for i in decollate_batch(main_output)]
        labels_post = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=outputs_post, y=labels_post)

    def on_validation_epoch_end(self):
        dice_metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log(
            "val_dice_metric", dice_metric, on_step=False, on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=0.99,
            nesterov=True,
            weight_decay=3e-5,
        )
        scheduler = PolyLRScheduler(
            optimizer=optimizer,
            initial_lr=self.learning_rate,
            max_steps=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=list_data_collate,
            shuffle=False,
        )
        return val_loader


if __name__ == "__main__":
    root_dir = os.environ.get("DATASET_ROOT", "./data")
    image_dir = os.path.join(root_dir, "imagesTr")
    label_dir = os.path.join(root_dir, "labelsTr")
    weight_dir = os.path.join(root_dir, "weights")
    splits_json = os.path.join(root_dir, "stratified_data_splits.json")
    cache_dir = os.path.join(root_dir, "cache_stage2")
    dataset_fingerprint_json = os.path.join(
        root_dir, "meta_info/dataset_fingerprint.json"
    )

    try:
        with open(dataset_fingerprint_json, "r") as f:
            dataset_fingerprint = json.load(f)
        mean_value = dataset_fingerprint["mean_foreground"]
        std_value = dataset_fingerprint["std_foreground"]
        clip_min_value = dataset_fingerprint["clip_lower_bound"]
        clip_max_value = dataset_fingerprint["clip_upper_bound"]
    except (FileNotFoundError, KeyError):
        mean_value = 0.0
        std_value = 1.0
        clip_min_value = -1000
        clip_max_value = 1000

    label_class_num = 5
    patch_size = (128, 128, 128)
    target_spacing = (1.0, 1.0, 1.0)
    train_patch_num = 500
    val_patch_num = 100
    seed = 42
    epoch_num = 150
    learning_rate = 1e-3
    device_index = 2
    fold_index = 0
    batch_size = 2
    num_workers = 12
    torch.set_float32_matmul_precision("medium")
    DEBUG = False

    experiment_name = "stage2"
    version_name = f"fold{fold_index}"
    log_dir = os.path.join(root_dir, "logs")
    tb_logger = pytorch_lightning.loggers.TensorBoardLogger(
        save_dir=log_dir, name=experiment_name, version=version_name
    )
    check_point_dir = os.path.join(
        root_dir,
        "checkpoints",
        experiment_name,
        version_name,
    )
    stage1_checkpoint_path = os.path.join(
        root_dir, f"checkpoints/stage1/fold{fold_index}/last.ckpt"
    )

    try:
        training_list_stage1, validation_list_stage1, test_list_stage1 = (
            split.load_split_and_get_datasets(
                json_path=splits_json,
                current_fold_idx=fold_index,
            )
        )
    except FileNotFoundError:
        training_list_stage1 = [
            {"image": "sample_image.nii.gz", "label": "sample_label.nii.gz"}
        ]
        validation_list_stage1 = [
            {"image": "sample_image.nii.gz", "label": "sample_label.nii.gz"}
        ]
        test_list_stage1 = [
            {"image": "sample_image.nii.gz", "label": "sample_label.nii.gz"}
        ]

    def add_weight_paths(item: dict, weight_directory: str) -> dict:
        series_uid = os.path.basename(item["label"]).removesuffix(".nii.gz")
        return {
            **item,
            "sample_weight": os.path.join(
                weight_directory, f"{series_uid}.sample_weight_prob.nii.gz"
            ),
            "loss_weight": os.path.join(
                weight_directory, f"{series_uid}.combined_loss_weight.nii.gz"
            ),
        }

    print("Processing training and validation lists to add weight paths...")
    training_list_stage2 = [
        add_weight_paths(item, weight_dir) for item in training_list_stage1
    ]
    validation_list_stage2 = [
        add_weight_paths(item, weight_dir) for item in validation_list_stage1
    ]

    if DEBUG:
        train_patch_num = 16
        val_patch_num = 8
        training_list_stage2 = training_list_stage2[0:4]
        validation_list_stage2 = validation_list_stage2[0:4]

    train_transforms = transforms.get_monai_transforms(
        mode="hard_case",
        sync_transform_keys=["image", "label", "sample_weight", "loss_weight"],
        mode_for_deformation=["bilinear", "nearest", "nearest", "nearest"],
        crop_foreground=False,
        patch_size=patch_size,
        target_spacing=target_spacing,
        num_samples=int(np.ceil(train_patch_num / len(training_list_stage2))),
        clip_min=clip_min_value,
        clip_max=clip_max_value,
        mean=mean_value,
        std=std_value,
    )
    val_transforms = transforms.get_monai_transforms(
        mode="hard_case_val",
        sync_transform_keys=["image", "label", "sample_weight", "loss_weight"],
        mode_for_deformation=["bilinear", "nearest", "nearest", "nearest"],
        patch_size=patch_size,
        target_spacing=target_spacing,
        num_samples=int(np.ceil(val_patch_num / len(validation_list_stage2))),
        clip_min=clip_min_value,
        clip_max=clip_max_value,
        mean=mean_value,
        std=std_value,
    )

    train_ds = PersistentDataset(
        data=training_list_stage2,
        transform=train_transforms,
        cache_dir=cache_dir,
    )
    val_ds = PersistentDataset(
        data=validation_list_stage2,
        transform=val_transforms,
        cache_dir=cache_dir,
    )

    net = Net(
        train_ds=train_ds,
        val_ds=val_ds,
        label_class_num=label_class_num,
        batch_size=batch_size,
        num_workers=num_workers,
        learning_rate=learning_rate,
    )

    try:
        stage1_checkpoint = torch.load(
            stage1_checkpoint_path, map_location=torch.device("cpu")
        )
        net.load_state_dict(stage1_checkpoint["state_dict"], strict=False)
        logging.info("Pretrained weights loaded")
    except FileNotFoundError:
        logging.info("Pretrained checkpoint not found, training from scratch")

    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        dirpath=check_point_dir,
        filename=None,
        monitor=None,
        save_weights_only=False,
        every_n_epochs=10,
        save_on_train_epoch_end=False,
        enable_version_counter=False,
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    last_ckpt_path = os.path.join(check_point_dir, "last.ckpt")
    if os.path.exists(last_ckpt_path):
        resume_from_checkpoint = last_ckpt_path
        logging.info(f"Resume training from checkpoint: {resume_from_checkpoint}")
    else:
        resume_from_checkpoint = None
        logging.info("Start training from scratch")

    trainer = pytorch_lightning.Trainer(
        devices=[device_index],
        max_epochs=epoch_num,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed",
    )

    trainer.fit(net, ckpt_path=resume_from_checkpoint)
