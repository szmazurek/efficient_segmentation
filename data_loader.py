import pandas as pd
from medpy.io import load

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


class SegmentationDataGenerator(Dataset):
    def __init__(self, data_path, img_size=224):
        self.data = pd.read_csv(data_path)
        self.img_size = img_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label_path = self.data.iloc[idx, 1]

        slice_idx = self.data.iloc[idx, 2]

        img_array = load(img_path)[0][:, :, slice_idx]
        label_array = load(label_path)[0][:, :, slice_idx]

        img_tensor = torch.from_numpy(img_array.astype(float)).float()
        label_tensor = torch.from_numpy(label_array.astype(float)).long()

        img_tensor = transforms.Resize(size=(self.img_size, self.img_size),
                                       interpolation=InterpolationMode.BILINEAR)(img_tensor.unsqueeze(0))
        label_tensor = transforms.Resize(size=(self.img_size, self.img_size),
                                         interpolation=InterpolationMode.NEAREST)(label_tensor.unsqueeze(0))

        return img_tensor, label_tensor[0]
