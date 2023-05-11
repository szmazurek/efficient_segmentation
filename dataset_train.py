import os
import pandas as pd
from medpy.io import load
import torchio as tio

from torch.utils.data import Dataset

from utils import verify_segmentation_dataset


class FetalBrainDataset(Dataset):
    def __init__(self, images_folder, masks_folder, img_size=224, transform=None):
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
        img_array_ = tio.ScalarImage(tensor=img_array.data[:, :, :, None, slice_idx].float(), affine=img_array.affine)
        label_array = tio.LabelMap(label_path)
        label_array_ = tio.LabelMap(tensor=label_array.data[:, :, :, None, slice_idx].long(), affine=img_array.affine)
        subject = tio.Subject(img=img_array_, label=label_array_)

        if self.transform:
            subject = self.transform(subject, img_size=self.img_size, intensity=True)

        # Get values from tio data
        img_data = subject['img']['data'].squeeze()
        label_data = subject['label']['data'].squeeze()

        img = img_data[None].float()
        label = label_data.long()

        return img, label


def create_dataset_csv(images_folder, masks_folder):
    images_paths = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
    masks_paths = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])

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

    path_df = pd.DataFrame({'image': x, 'mask': y, 'slice': index})
    return path_df


def preprocess(input_tio, img_size=224, intensity=False):
    target_spacing = (input_tio.shape[1] * input_tio.spacing[0] / (img_size - 1),
                      input_tio.shape[1] * input_tio.spacing[1] / (img_size - 1),
                      1)

    if intensity:
        prep_transform = tio.Compose([
            tio.Resample(target_spacing),
            tio.CropOrPad((img_size, img_size, input_tio.shape[-1])),
            tio.RescaleIntensity(out_min_max=(0, 1.0), percentiles=(0.5, 99.5))
        ])
    else:
        prep_transform = tio.Compose([
            tio.Resample(target_spacing),
            tio.CropOrPad((img_size, img_size, input_tio.shape[-1])),
        ])

    # Apply preprocessing to images and segmentation maps
    preprocessed_tio = prep_transform(input_tio)

    return preprocessed_tio
