import torch
import lightning.pytorch as pl
from torchmetrics import Dice
from monai.data import decollate_batch
from monai.optimizers import Novograd
from monai.transforms import Compose
from monai.inferers import SliceInferer
from utils.utils import return_chosen_model, return_chosen_loss


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model="Unet",
        loss="DiceLoss",
        in_channels=1,
        init_features=64,
        lr=1e-3,
        in_shape=(None, 1, 256, 256),
        predict_transforms: Compose = None,
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
        x = batch["image"]
        inferer = SliceInferer(
            roi_size=(self.in_shape[2], self.in_shape[3]),
            spatial_dim=2,
            progress=False,
        )
        y_hat = inferer(x, self.model)
        # #### THIS DOES NOT WORK ON OPENNEURO
        # THIS IS DUE TO SOME PROBLEMS WITH INVERSION OF
        # AFFINE TRANSFORMS. THIS VERSION RUNS ON 3D VOLUMES ONLY FROM
        # THE DATASET PROVIDED BY THE CHALLENGE.
        # #####
        batch_copied = batch.copy()
        batch_copied["pred"] = y_hat
        batch_copied = [
            self.predict_transforms(i) for i in decollate_batch(batch_copied)
        ]
        return y_hat

    def configure_optimizers(self):
        optimizer = Novograd(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
            # weight_decay=0.001,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=5,
            verbose=True,
            threshold=0.001,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
