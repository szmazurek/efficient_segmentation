import SimpleITK as sitk
import numpy as np
import os
import torch
import pandas as pd

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode



def test_model(epoch, model, data_path, device, save_flag=True):
    # Set model to eval mode
    model.eval()

    data = pd.read_csv(data_path)

    # Iterate over test data
    with torch.no_grad():
        for test_data_idx in range(len(data)):
            img_path = data.iloc[test_data_idx, 0]
            image = sitk.ReadImage(img_path)

            inputs = sitk.GetArrayFromImage(image)
            org_shape = inputs.shape
            prediction = np.zeros_like(inputs)

            inputs = torch.from_numpy(inputs.astype(float))[:, None].float().to(device)
            inputs = transforms.Resize(size=(224, 224),
                                       interpolation=InterpolationMode.NEAREST)(inputs)

            # Forward pass
            for ind in range(inputs.shape[0]):
                input_slice = inputs[ind].unsqueeze(0)
                outputs = model(input_slice)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                out = transforms.Resize(size=(org_shape[1], org_shape[2]),
                                        interpolation=InterpolationMode.NEAREST)(out)
                out = out.squeeze(0).cpu().detach().numpy()
                prediction[ind] = out

        if save_flag:
            save_path = os.path.join(os.path.dirname(img_path), 'predicted_masks_epoch' + str(epoch))
            os.makedirs(save_path, exist_ok=True)
            save_name = 'mask' + os.path.basename(img_path)

            result_image = sitk.GetImageFromArray(prediction)
            result_image.CopyInformation(image)

            # write the image
            sitk.WriteImage(result_image, os.path.join(save_path, save_name))

    return "Testing Finished!"
