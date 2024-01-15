import lightning.pytorch as pl
import torch
from monai.inferers import SliceInferer
from monai.optimizers import Novograd
from monai.transforms import Compose
from torchmetrics import Dice
from utils.utils import return_chosen_loss, return_chosen_model


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model="Unet",
        loss="DiceLoss",
        in_channels=1,
        init_features=64,
        lr=1e-3,
        in_shape=(None, 1, 256, 256),
        predict_transforms: Compose | None = None,
        save_dir="data/dummy_results",
    ):
        super().__init__()

        self.model = return_chosen_model(
            model, in_channels, in_shape, init_features
        )
        self.loss = return_chosen_loss(loss)

        self.dice_score = Dice(multiclass=False, average="micro")
        self.in_shape = in_shape
        self.lr = lr
        self.predict_transforms = predict_transforms
        self.save_dir = save_dir

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]

        y_hat = self(x)
        loss = self.loss(y_hat.float(), y.float())
        dice_score = self.dice_score(y_hat, y.int())
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

        y_hat = self(x)

        loss = self.loss(y_hat.float(), y.float())
        dice_score = self.dice_score(y_hat, y.int())
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

        y_hat = self(x)
        loss = self.loss(y_hat.float(), y.float())
        dice_score = self.dice_score(y_hat, y.int())
        self.log(
            "test_dice_score",
            dice_score,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            # logger=True,
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
        x = batch["image"]
        inferer = SliceInferer(
            roi_size=(self.in_shape[2], self.in_shape[3]),
            spatial_dim=2,
            progress=False,
        )
        y_hat = {}
        y_hat["image"] = x[0]
        y_hat["pred"] = inferer(x, self.model)[0]
        self.predict_transforms(y_hat)
        return y_hat

    def configure_optimizers(self):
        optimizer = Novograd(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True,
            threshold=0.001,
            min_lr=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
