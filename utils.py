import os
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, n_classes, batch=False, focal=False):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes
        self.batch = batch

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        dims = (0, 1, 2) if self.batch else (1, 2)

        intersect = torch.sum(score * target, dims)
        y_sum = torch.sum(target * target, dims)
        z_sum = torch.sum(score * score, dims)
        dice_score = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1.0 - dice_score if self.batch else torch.mean(1.0 - dice_score)
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), \
            'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def verify_segmentation_dataset(images_list, masks_list):
    assert len(images_list) == len(masks_list), \
        "Found error during data loading: number of images and number of masks do not match"

    for item in range(len(images_list)):
        name = os.path.basename(images_list[item])
        mask_name = os.path.join(os.path.dirname(masks_list[item]), 'mask' + name[3:])

        assert mask_name in masks_list, \
            "Found error during data loading: No mask was found for\n{0}".format(str(name))
