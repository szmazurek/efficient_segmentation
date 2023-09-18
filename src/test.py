import os
import warnings
from glob import glob

import lightning.pytorch as pl
import numpy as np
import torch
from models.lightning_module import LightningModel
from monai.data import DataLoader, Dataset
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    MapTransform,
    RemoveSmallObjectsd,
    ResizeWithPadOrCropd,
    SaveImaged,
    Spacingd,
)

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")


class SliceWiseNormalizeIntensityd(MapTransform):
    def __init__(self, keys, subtrahend=0.0, divisor=None, nonzero=True):
        super().__init__(keys)
        self.subtrahend = subtrahend
        self.divisor = divisor
        self.nonzero = nonzero

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            for i in range(image.shape[-1]):
                slice_ = image[..., i]
                if self.nonzero:
                    mask = slice_ > 0
                    if np.any(mask):
                        if self.subtrahend is None:
                            slice_[mask] = slice_[mask] - slice_[mask].mean()
                        else:
                            slice_[mask] = slice_[mask] - self.subtrahend

                        if self.divisor is None:
                            slice_[mask] /= slice_[mask].std()
                        else:
                            slice_[mask] /= self.divisor

                else:
                    if self.subtrahend is None:
                        slice_ = slice_ - slice_.mean()
                    else:
                        slice_ = slice_ - self.subtrahend

                    if self.divisor is None:
                        slice_ /= slice_.std()
                    else:
                        slice_ /= self.divisor

                image[..., i] = slice_
            d[key] = image
        return d


def test_lightning(args):
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys="image", pixdim=(1.0, 1.0, -1.0), mode="bilinear"),
            SliceWiseNormalizeIntensityd(keys=["image"], nonzero=True),
            ResizeWithPadOrCropd(
                keys="image", spatial_size=(args.img_size, args.img_size, -1)
            ),
        ]
    )

    post_transforms = Compose(
        [
            Invertd(
                keys="pred",
                transform=test_transforms,
                orig_keys="image",
                meta_keys="pred_meta_dict",
                orig_meta_keys="image_meta_dict",
                meta_key_postfix="meta_dict",
                nearest_interp=False,
                to_tensor=True,
            ),
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True, to_onehot=None),
            RemoveSmallObjectsd(keys="pred", min_size=50, connectivity=1),
            SaveImaged(
                keys="pred",
                meta_keys="pred_meta_dict",
                output_dir=args.test_results_save_path,
                separate_folder=False,
                output_postfix="maskpred",
                resample=False,
            ),
        ]
    )

    # load data
    test_images = sorted(
        glob(os.path.join(args.testing_data_path, "*.nii.gz"))
    )
    test_files = [{"image": image_name} for image_name in test_images]

    test_dataset = Dataset(data=test_files, transform=test_transforms)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=6,
        prefetch_factor=10,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    model = LightningModel.load_from_checkpoint(
        args.model_path,
        in_shape=(None, 1, args.img_size, args.img_size),
        loss=args.loss_function,
        model=args.model,
        save_dir=args.test_results_save_path,
        predict_transforms=post_transforms,
    )
    strategy = pl.strategies.DDPStrategy(
        find_unused_parameters=False,
        static_graph=True,
    )
    if not os.path.exists("lightning_logs"):
        os.mkdir("lightning_logs")
    trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        num_nodes=1,
        precision="16-mixed",
        strategy=strategy,
    )
    trainer.predict(model, test_dataloader)
