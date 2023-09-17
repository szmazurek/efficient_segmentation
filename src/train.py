import os
import random
import warnings
from glob import glob

import lightning.pytorch as pl
import monai
import monai.transforms as tr
import torch
import wandb
import numpy as np
from codecarbon import OfflineEmissionsTracker

from models.lightning_module import LightningModel
from monai.data import (
    DataLoader,
    CacheDataset,
    partition_dataset,
)

from utils.dataloader_utils import per_patient_split
from utils.utils import AVAILABLE_MODELS
import logging

torch.set_float32_matmul_precision("medium")


for name in logging.Logger.manager.loggerDict.keys():
    if "codecarbon" in name:
        logging.getLogger(name).setLevel(logging.CRITICAL)


pl.seed_everything(42)
N_PROC = int(os.environ["SLURM_NTASKS"])


def train_lightning(args):
    images = sorted(
        glob(os.path.join(args.training_data_path, "images/*.nii.gz"))
    )
    labels = sorted(
        glob(os.path.join(args.training_data_path, "masks/*.nii.gz"))
    )

    files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(images, labels)
    ]

    # define transforms for image and segmentation
    transformations = tr.Compose(
        [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, -1.0),
                mode=("bilinear", "nearest"),
            ),
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
            tr.NormalizeIntensityd(keys=["image"], nonzero=True),
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(args.img_size, args.img_size),
            ),
            tr.RandFlipd(keys=["image", "label"], prob=0.3),
            tr.RandRotated(keys=["image", "label"], range_x=90),
        ]
    )

    transformations_val_test = tr.Compose(
        [
            tr.LoadImaged(keys=["image", "label"]),
            tr.EnsureChannelFirstd(keys=["image", "label"]),
            tr.Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, -1.0),
                mode=("bilinear", "nearest"),
            ),
            tr.SqueezeDimd(keys=["image", "label"], dim=-1, update_meta=True),
            tr.NormalizeIntensityd(keys=["image"], nonzero=True),
            tr.ResizeWithPadOrCropd(
                keys=["image", "label"],
                spatial_size=(args.img_size, args.img_size),
            ),
        ]
    )

    random.shuffle(files)
    files = np.array(files)

    train_files, val_files, test_files = per_patient_split(files)

    train_data_partitioned = partition_dataset(
        data=train_files,
        num_partitions=N_PROC,
        shuffle=True,
        even_divisible=False,
        seed=42,
    )[int(os.environ["SLURM_PROCID"])]

    val_data_partitioned = partition_dataset(
        data=val_files,
        num_partitions=N_PROC,
        shuffle=True,
        even_divisible=False,
        seed=42,
    )[int(os.environ["SLURM_PROCID"])]

    test_data_partitioned = partition_dataset(
        data=test_files,
        num_partitions=N_PROC,
        seed=42,
        shuffle=False,
        even_divisible=False,
    )[int(os.environ["SLURM_PROCID"])]

    train_dataset = CacheDataset(
        data=train_data_partitioned,
        transform=transformations,
        num_workers=30,
        # cache_rate=0.01,
    )
    val_dataset = CacheDataset(
        data=val_data_partitioned,
        transform=transformations_val_test,
        num_workers=30,
        # cache_rate=0.01,
    )

    test_dataset = CacheDataset(
        test_data_partitioned,
        transform=transformations_val_test,
        num_workers=args.n_workers,
        cache_rate=0.01,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.n_workers,
        pin_memory=True,
        prefetch_factor=20,
        drop_last=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        prefetch_factor=5,
        drop_last=False,
        persistent_workers=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_workers,
        pin_memory=True,
        prefetch_factor=1,
        drop_last=False,
        persistent_workers=False,
    )

    if args.wandb:
        if int(os.environ["SLURM_PROCID"]) == 0:
            wandb.init(
                entity="mazurek",
                project="E2MIP_Challenge_FetalBrainSegmentation",
                group="all-models-eval-final",
                name=f"{args.exp_name}",
            )
        wandb_logger = pl.loggers.WandbLogger(
            name=args.exp_name,
        )

    model = LightningModel(
        loss=args.loss_function,
        model=args.model,
        in_shape=(None, 1, args.img_size, args.img_size),
        lr=args.lr,
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1, mode="min", monitor="val_loss"
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        mode="min",
        verbose=True,
        min_delta=0.001,
    )

    lr_finder_callback = pl.callbacks.LearningRateFinder(
        min_lr=1e-5,
        max_lr=1e-1,
        num_training_steps=100,
    )

    strategy = pl.strategies.DDPStrategy(
        find_unused_parameters=False,
        static_graph=True,
    )

    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        precision="16-mixed",
        strategy=strategy,
        num_nodes=1,
        enable_model_summary=True,
        max_epochs=args.epochs,
        logger=wandb_logger if args.wandb else None,
        callbacks=[
            model_checkpoint_callback,
            early_stopping_callback,
            lr_finder_callback,
        ],
        log_every_n_steps=1,
        num_sanity_val_steps=0,
    )
    tracker = OfflineEmissionsTracker(
        country_iso_code="POL",
        # output_dir="output_files/codecarbon",
        # output_file=f"{datetime.datetime.now()}.csv",
    )
    tracker.start()
    trainer.fit(model, train_dataloader, val_loader)

    tracker.stop()
    energy_training = round(tracker._total_energy.kWh * 3600, 3)
    tracker = OfflineEmissionsTracker(
        country_iso_code="POL",
        # output_dir="output_files/codecarbon",
        # output_file=f"{datetime.datetime.now()}.csv",
    )
    tracker.start()

    results = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    tracker.stop()
    energy_inference = round(tracker._total_energy.kWh * 3600, 3)
    training_efficiency_measure = (
        results[0]["test_dice_score"] * 100 - energy_training
    )
    total_efficiency_measure = (
        results[0]["test_dice_score"] * 100
        - energy_training
        - energy_inference
    )

    if args.wandb and int(os.environ["SLURM_PROCID"]) == 0:
        wandb.log(
            {
                "energy_training_kJ": energy_training,
                "energy_inference_kJ": energy_inference,
                "energy_total_kJ": energy_training + energy_inference,
                "training_efficiency_measure": training_efficiency_measure,
                "total_efficiency_measure": total_efficiency_measure,
            }
        )
        wandb.finish()
    print("Finished training.")
