import os
from glob import glob

import lightning.pytorch as pl
import monai.transforms as tr
import torch
import wandb
from models.lightning_module import LightningModel
from monai.data import CacheDataset, DataLoader, partition_dataset

torch.set_float32_matmul_precision("medium")


pl.seed_everything(42)
N_PROC = torch.cuda.device_count()


def train_lightning(args):
    train_images = sorted(
        glob(os.path.join(args.training_data_path, "images/*.nii.gz"))
    )
    train_labels = sorted(
        glob(os.path.join(args.training_data_path, "masks/*.nii.gz"))
    )
    train_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]

    validation_images = sorted(
        glob(os.path.join(args.validation_data_path, "images/*.nii.gz"))
    )
    validation_labels = sorted(
        glob(os.path.join(args.validation_data_path, "masks/*.nii.gz"))
    )
    validation_files = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(validation_images, validation_labels)
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
    print("Global rank:")
    global_rank = int(trainer.global_rank)
    print(global_rank)
    train_data_partitioned = partition_dataset(
        data=train_files,
        num_partitions=N_PROC,
        shuffle=True,
        even_divisible=False,
        seed=42,
    )[global_rank]

    val_data_partitioned = partition_dataset(
        data=validation_files,
        num_partitions=N_PROC,
        shuffle=True,
        even_divisible=False,
        seed=42,
    )[global_rank]

    train_dataset = CacheDataset(
        data=train_data_partitioned,
        transform=transformations,
        num_workers=30,
    )
    val_dataset = CacheDataset(
        data=val_data_partitioned,
        transform=transformations_val_test,
        num_workers=30,
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

    trainer.fit(model, train_dataloader, val_loader)
    wandb.finish()
    print("Finished training.")
