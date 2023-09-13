import os
import pandas as pd
from medpy.io import load
import torchio as tio
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from .utils import (
    verify_segmentation_dataset,
)


class FetalBrainDataset(Dataset):
    def __init__(
        self, images_folder, masks_folder, img_size=224, transform=None
    ):
        self.data = create_dataset_csv(images_folder, masks_folder)
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label_path = self.data.iloc[idx, 1]
        slice_idx = self.data.iloc[idx, 2]

        # Create TorchIO data
        img_array = tio.ScalarImage(img_path)
        img_array_ = tio.ScalarImage(
            tensor=img_array.data[:, :, :, None, slice_idx].float(),
            affine=img_array.affine,
        )
        label_array = tio.LabelMap(label_path)
        label_array_ = tio.LabelMap(
            tensor=label_array.data[:, :, :, None, slice_idx].long(),
            affine=img_array.affine,
        )
        subject = tio.Subject(img=img_array_, label=label_array_)

        if self.transform:
            subject = self.transform(
                subject, img_size=self.img_size, intensity=True
            )

        # Get values from tio data
        img_data = subject["img"]["data"].squeeze()
        label_data = subject["label"]["data"].squeeze()

        img = img_data[None].float()
        label = label_data.long()

        return img, label


def create_dataset_csv(images_folder, masks_folder):
    images_paths = sorted(
        [os.path.join(images_folder, f) for f in os.listdir(images_folder)]
    )
    masks_paths = sorted(
        [os.path.join(masks_folder, f) for f in os.listdir(masks_folder)]
    )

    verify_segmentation_dataset(images_paths, masks_paths)

    _paths = []
    x = []
    y = []
    index = []
    for i in range(len(images_paths)):
        image_path = images_paths[i]
        mask_path = masks_paths[i]

        mri, _ = load(image_path)
        for slice_idx in range(mri.shape[2]):
            _paths.append([image_path, mask_path, slice_idx])

    # random.shuffle(_paths)
    for features, label, k in _paths:
        x.append(features)
        y.append(label)
        index.append(k)

    path_df = pd.DataFrame({"image": x, "mask": y, "slice": index})
    return path_df


def preprocess(input_tio, img_size=224, intensity=False):
    target_spacing = (
        input_tio.shape[1] * input_tio.spacing[0] / (img_size - 1),
        input_tio.shape[1] * input_tio.spacing[1] / (img_size - 1),
        1,
    )

    if intensity:
        prep_transform = tio.Compose(
            [
                tio.Resample(target_spacing),
                tio.CropOrPad((img_size, img_size, input_tio.shape[-1])),
                tio.RescaleIntensity(
                    out_min_max=(0, 1.0), percentiles=(0.5, 99.5)
                ),
            ]
        )
    else:
        prep_transform = tio.Compose(
            [
                tio.Resample(target_spacing),
                tio.CropOrPad((img_size, img_size, input_tio.shape[-1])),
            ]
        )

    # Apply preprocessing to images and segmentation maps
    preprocessed_tio = prep_transform(input_tio)

    return preprocessed_tio


def per_patient_split(
    data_list: np.ndarray,
    test_percentage: float = 0.1,
    val_percentage: float = 0.1,
    seed: int = 42,
):
    """Performs train/val/test split on the data_list based on the patient
    ID in the filename. All data from a subject is assigned to the same split,
    so there is no data leakage between splits. It first separates
    test_percentage of the subjects for testing, then separates val_percentage
    of the remaining subjects for validation. The rest are used for training.
    Args:
        data_list (np.ndarray): List of dictionaries with keys "image" and
            "mask" containing the filenames of the images and masks
        test_percentage (float, optional): Percentage of subjects to use for
            testing. Defaults to 0.1.
        val_percentage (float, optional): Percentage of subjects to use for
            validation. Defaults to 0.1.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        train_data (np.ndarray): List of filenames and labels for training
        val_data (np.ndarray): List of filenames and labels for validation
        test_data (np.ndarray): List of filenames and labels for testing
    """

    unique_subjects = np.unique(
        [
            os.path.basename(fname_dict["image"]).split("_")[1]
            for fname_dict in data_list
        ]
    )

    train_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_percentage, random_state=seed
    )
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=val_percentage, random_state=seed
    )

    train_mask = np.array(
        [
            os.path.basename(fname_dict["image"]).split("_")[1]
            in train_subjects
            for fname_dict in data_list
        ]
    )
    val_mask = np.array(
        [
            os.path.basename(fname_dict["image"]).split("_")[1] in val_subjects
            for fname_dict in data_list
        ]
    )
    test_mask = np.array(
        [
            os.path.basename(fname_dict["image"]).split("_")[1]
            in test_subjects
            for fname_dict in data_list
        ]
    )
    train_data = data_list[train_mask]
    val_data = data_list[val_mask]
    test_data = data_list[test_mask]

    return train_data, val_data, test_data
