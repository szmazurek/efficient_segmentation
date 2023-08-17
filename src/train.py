import os
import random
import warnings
from glob import glob

import lightning.pytorch as pl
import monai
import monai.transforms as tr
import torch

from lightning_bagua import BaguaStrategy
from models.lightning_module import LightningModel
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.inferers import SimpleInferer
from monai.metrics import DiceMetric
import time

# warnings.filterwarnings("ignore")

pl.seed_everything(42)


def train(args):
    monai.config.print_config()

    # load data
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
        ]
    )

    # create validation
    random.shuffle(files)
    n_split = int(0.8 * len(files))

    train_ds = Dataset(data=files[:n_split], transform=transformations)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    val_ds = Dataset(data=files[-n_split:], transform=transformations)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=4)

    # define metrics
    dice_metric = DiceMetric(
        include_background=False, reduction="mean", get_not_nans=False
    )
    post_pred = tr.Compose(
        [tr.AsDiscrete(argmax=True, to_onehot=args.num_classes)]
    )
    post_label = tr.Compose([tr.AsDiscrete(to_onehot=args.num_classes)])

    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
    ).to(device)
    loss_function = monai.losses.DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
        squared_pred=False,
        batch=True,
        smooth_nr=0.00001,
        smooth_dr=0.00001,
        lambda_dice=0.6,
        lambda_ce=0.4,
    )
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    # start a typical PyTorch training
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    for epoch in range(args.epochs):
        print("-" * 30)
        print(f"epoch {epoch + 1}/{args.epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    inferer = SimpleInferer()
                    val_outputs = inferer(val_inputs, model)
                    val_outputs = [
                        post_pred(i) for i in decollate_batch(val_outputs)
                    ]
                    val_labels = [
                        post_label(i) for i in decollate_batch(val_labels)
                    ]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()
                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(
                        model.state_dict(),
                        "best_metric_model_segmentation2d.pth",
                    )
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}"
    )


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
        ]
    )

    # cache works and works good for single process training. It seems that
    # when using runtime process caching, everything is fine. Changing cache rate does
    # not seem to affect the GPU memory allocation when in this mode.
    # When all the data is preloaded into cache before starting training,
    # the GPU memory exploded, but dunno why - now it works *LOL*.

    main_dataset = CacheDataset(
        data=files,
        transform=transformations,
        num_workers=4,
        # cache_rate=0.5,
        # runtime_cache="processes",
    )
    # main_dataset = Dataset(
    #     data=files,
    #     transform=transformations,
    # )

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        main_dataset, [0.8, 0.1, 0.1]
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
        persistent_workers=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
        persistent_workers=False,
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
        monitor="val_loss", patience=15, mode="min", verbose=True
    )
    if args.wandb:
        wandb_logger = pl.loggers.WandbLogger(
            project="E2MIP_Challenge_FetalBrainSegmentation",
            entity="mazurek",
            log_model=False,
            name=args.exp_name,
        )
    visible_devices = len(os.environ["CUDA_VISIBLE_DEVICES"])
    # strategy = pl.strategies.DDPStrategy(find_unused_parameters=False)
    # pl.strategies.SingleDeviceStrategy(device=args.device)
    # if visible_devices == 1
    # else
    print(f"Using {visible_devices} devices for training.")
    torch.set_float32_matmul_precision("medium")
    strategy = BaguaStrategy(
        algorithm="bytegrad",
    )
    trainer = pl.Trainer(
        devices="auto",
        accelerator="gpu",
        precision="16-mixed",
        strategy=strategy,
        num_nodes=1,
        enable_model_summary=False,
        max_epochs=args.epochs,
        logger=wandb_logger if args.wandb else None,
        callbacks=[model_checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
    )
    start = time.time()
    trainer.fit(model, train_dataloader, val_loader)
    print(f"Training took {time.time() - start} seconds.")
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")
    print("Finished training.")
