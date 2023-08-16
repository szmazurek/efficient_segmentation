import os
import torch
import lightning.pytorch as pl

# from network import UNet, LightningUnet
from models.lightning_module import LightningModel
from utils.dataloader_utils import (
    FetalBrainDataset,
    preprocess,
)
from utils.utils import DiceLoss


def train(args):
    # Set up dataset and data loaders
    images_folder = os.path.join(args.training_data_path, "images")
    masks_folder = os.path.join(args.training_data_path, "masks")
    train_dataset = FetalBrainDataset(
        images_folder, masks_folder, img_size=224, transform=preprocess
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    # Set up device
    device = torch.device(args.device)

    # Set up model, optimizer, and criterion
    model = LightningModel(in_shape=(None, 1, 224, 224)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = DiceLoss(n_classes=args.num_classes)

    # Train for multiple epochs
    print("Initializing training...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets, softmax=True)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print(f"Completed batch {i+1} of epoch {epoch+1}")
        # Print training and test metrics
        print(f"Epoch {epoch}: train_loss = {running_loss / i+1:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(), f"model_checkpoint_epoch_{epoch}.pt")


def train_lightning(args):
    images_folder = os.path.join(args.training_data_path, "images")
    masks_folder = os.path.join(args.training_data_path, "masks")
    main_dataset = FetalBrainDataset(
        images_folder, masks_folder, img_size=224, transform=preprocess
    )

    train_dataset, val_dataset = torch.utils.data.random_split(
        main_dataset, [0.9, 0.1]
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False,
    )
    model = LightningModel(
        loss=args.loss_function,
        model=args.model,
        in_shape=(None, 1, 224, 224),
        lr=args.lr,
    )
    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1, mode="min", monitor="val_loss"
    )
    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, mode="min", verbose=True
    )
    if args.wandb:
        api_key = open("wandb_api_key.txt", "r")
        key = api_key.read()
        api_key.close()
        os.environ["WANDB_API_KEY"] = key
        wandb_logger = pl.loggers.WandbLogger(
            project="E2MIP_Challenge_FetalBrainSegmentation",
            entity="mazurek",
            log_model=False,
        )
    visible_devices = len(os.environ["CUDA_VISIBLE_DEVICES"])
    strategy = (
        pl.strategies.SingleDeviceStrategy(device=args.device)
        if visible_devices == 1
        else pl.strategies.DDPStrategy(find_unused_parameters=False)
    )
    print(f"Using {visible_devices} devices for training.")
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        devices="auto",
        precision="16-mixed",
        strategy=strategy,
        enable_model_summary=False,
        max_epochs=args.epochs,
        logger=wandb_logger if args.wandb else None,
        callbacks=[model_checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
        sync_batchnorm=True,
    )

    trainer.fit(model, train_dataloader, val_loader)
    print("Finished training.")
