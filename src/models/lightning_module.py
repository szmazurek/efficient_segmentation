import lightning.pytorch as pl
import torch
from torchmetrics import Dice
from monai.metrics import DiceMetric
import segmentation_models_pytorch as smp
from .torch_models import UNet, AttSqueezeUNet


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model="Unet",
        loss="DiceLoss",
        in_channels=1,
        out_channels=2,
        init_features=64,
        lr=1e-3,
        in_shape=(None, 1, 256, 256),
    ):
        super().__init__()

        if model == "Unet":
            self.model = UNet(in_channels, in_channels, init_features)
        elif model == "AttSqueezeUnet":
            self.model = AttSqueezeUNet(1, in_shape)
        if loss == "DiceLoss":
            self.loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
        elif loss == "MCCLoss":
            self.loss = smp.losses.MCCLoss()
        self.dice_score = Dice(multiclass=False, average="micro")
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y = y.int()
        y_hat = self(x)

        loss = self.loss(y_hat, y)
        dice_score = self.dice_score(y_hat, y)
        self.log(
            "dice_score",
            dice_score,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )  # disable for final version

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y = y.int()
        y_hat = self(x)

        loss = self.loss(y_hat, y)

        dice_score = self.dice_score(y_hat, y)
        self.log(
            "val_dice_score",
            dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y = y.int()
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        dice_score = self.dice_score(y_hat, y)
        self.log(
            "test_dice_score",
            dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        y_hat = self(x)
        return y_hat

    # def save_prediction_batch(self,result_mask):

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
