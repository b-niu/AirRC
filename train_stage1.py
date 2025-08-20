import logging
import os
import sys
import json

import numpy as np
import pytorch_lightning
import torch
from monai.data import DataLoader, PersistentDataset, decollate_batch, list_data_collate
from monai.metrics import DiceMetric
from monai.networks.nets import DynUNet
from monai.transforms import AsDiscrete, Compose, EnsureType
from pytorch_lightning.callbacks import LearningRateMonitor

sys.path.append(".")
import config_optimizer
import dataset
import split
from config_loss import DeepSupervisionDiceCELoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Net(pytorch_lightning.LightningModule):
    def __init__(
        self,
        train_ds,
        val_ds,
        label_class_num=5,
        batch_size=4,
        num_workers=8,
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
            norm_name=("INSTANCE", {"eps": 1e-05, "affine": True}),
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

        self.loss_function = DeepSupervisionDiceCELoss(deep_supr_num=4)
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

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        outputs = self.forward(images)
        loss = self.loss_function(outputs, labels)
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
        optimizer = config_optimizer.configure_SGD_optimizers(self)
        return optimizer

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
    splits_json = os.path.join(root_dir, "stratified_data_splits.json")
    cache_dir = os.path.join(root_dir, "cache")
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
    device_index = 0
    fold_index = 0
    batch_size = 2
    num_workers = 16
    torch.set_float32_matmul_precision("medium")
    DEBUG = False

    experiment_name = "stage1"
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

    try:
        training_list, validation_list, test_list = split.load_split_and_get_datasets(
            json_path=splits_json,
            current_fold_idx=fold_index,
        )
    except FileNotFoundError:
        training_list = [
            {"image": "sample_image.nii.gz", "label": "sample_label.nii.gz"}
        ]
        validation_list = [
            {"image": "sample_image.nii.gz", "label": "sample_label.nii.gz"}
        ]
        test_list = [{"image": "sample_image.nii.gz", "label": "sample_label.nii.gz"}]

    if DEBUG:
        train_patch_num = 16
        val_patch_num = 8
        training_list = training_list[0:4]
        validation_list = validation_list[0:4]

    train_transforms = dataset.get_monai_transforms(
        mode="train",
        keys=["image", "label"],
        patch_size=patch_size,
        target_spacing=target_spacing,
        num_samples=int(np.ceil(train_patch_num / len(training_list))),
        clip_min=clip_min_value,
        clip_max=clip_max_value,
        mean=mean_value,
        std=std_value,
    )
    val_transforms = dataset.get_monai_transforms(
        mode="val",
        keys=["image", "label"],
        patch_size=patch_size,
        target_spacing=target_spacing,
        num_samples=int(np.ceil(val_patch_num / len(validation_list))),
        clip_min=clip_min_value,
        clip_max=clip_max_value,
        mean=mean_value,
        std=std_value,
    )

    train_ds = PersistentDataset(
        data=training_list,
        transform=train_transforms,
        cache_dir=cache_dir,
    )
    val_ds = PersistentDataset(
        data=validation_list,
        transform=val_transforms,
        cache_dir=cache_dir,
    )

    net = Net(
        train_ds=train_ds,
        val_ds=val_ds,
        label_class_num=label_class_num,
        batch_size=batch_size,
        num_workers=num_workers,
    )

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
        max_epochs=1000,
        logger=tb_logger,
        enable_checkpointing=True,
        num_sanity_val_steps=1,
        log_every_n_steps=5,
        callbacks=[checkpoint_callback, lr_monitor],
        precision="16-mixed",
    )

    trainer.fit(net, ckpt_path=resume_from_checkpoint)
