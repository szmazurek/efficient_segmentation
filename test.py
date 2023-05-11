import os
import time
import numpy as np

import torch
import torchio as tio

from network import UNet
from dataset_train import preprocess
from utils import verify_segmentation_dataset


def test_one_volume(model, input_volume_path, device):
    image = tio.ScalarImage(input_volume_path)
    prediction = np.zeros(image.shape[1:])

    for test_data_idx in range(image.shape[-1]):
        img_array = tio.ScalarImage(tensor=image.data[:, :, :, None, test_data_idx].float(),
                                    affine=image.affine)

        img_array = preprocess(img_array, img_size=224, intensity=True)
        inputs = img_array['data'][:, None, :, :, 0].to(device)

        outputs = model(inputs)
        outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

        label_array = tio.LabelMap(tensor=outputs[:, :, :, None].cpu().long(), affine=img_array.affine)
        label_array = preprocess(label_array, img_size=image.shape[1], intensity=False)

        out = label_array['data'].squeeze().cpu().detach().numpy()
        prediction[:, :, test_data_idx] = out

    return prediction


def test(args):
    # Set up dataset
    images_folder = os.path.join(args.testing_data_path, "images")
    masks_folder = os.path.join(args.testing_data_path, "masks")
    images_paths = sorted([os.path.join(images_folder, f) for f in os.listdir(images_folder)])
    masks_paths = sorted([os.path.join(masks_folder, f) for f in os.listdir(masks_folder)])
    verify_segmentation_dataset(images_paths, masks_paths)

    assert len(images_paths) == len(masks_paths), \
        'number of images and number of masks do not match'.format(len(images_paths), len(masks_paths))

    # Load the model
    model = UNet().to(args.device)
    msg = model.load_state_dict(torch.load(args.checkpoint_path))
    print("model loaded", msg)

    # Set model to eval mode
    model.eval()

    # Iterate over test data
    with torch.no_grad():
        for idx in range(len(images_paths)):
            predicted_mask = test_one_volume(model=model, input_volume_path=images_paths[idx], device=args.device)
            actual_mask = tio.LabelMap(masks_paths[idx])

        if args.save_path:
            os.makedirs(args.save_path, exist_ok=True)
            save_name = os.path.basename(masks_paths[idx])

            result_mask = tio.LabelMap(tensor=predicted_mask[None], affine=actual_mask.affine)

            # write the image
            result_mask.save(os.path.join(args.save_path, save_name))

    return "Testing Finished!"
