import lightning.pytorch as pl
from torchmetrics import Dice
import segmentation_models_pytorch as smp
from monai.data import decollate_batch
from .torch_models import UNet, AttSqueezeUNet, MobileNetV3Seg, MicroNet
from monai.optimizers import Novograd
from monai.transforms import Compose
from monai.inferers import SliceInferer


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

        if model == "Unet":
            self.model = UNet(in_channels, in_channels, init_features)
        elif model == "AttSqueezeUnet":
            self.model = AttSqueezeUNet(1, in_shape)
        elif model == "UnetSMP":
            self.model = smp.MAnet(
                encoder_name="efficientnet-b0",
                classes=1,
                in_channels=1,
                encoder_weights=None,
                activation="sigmoid",
            )
        elif model == "MobileNetV3":
            self.model = MobileNetV3Seg(nclass=1, pretrained_base=False)
        elif model == "MicroNet":
            self.model = MicroNet(nb_classes=1, inputs_shape=in_shape[1:])

        if loss == "DiceLoss":
            self.loss = smp.losses.DiceLoss(mode="binary", from_logits=False)
        elif loss == "MCCLoss":
            self.loss = smp.losses.MCCLoss()

        self.dice_score = Dice(multiclass=False, average="micro")
        self.in_shape = in_shape
        self.lr = lr
        self.predict_transforms = predict_transforms
        self.save_dir = save_dir

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
        )

        return optimizer
