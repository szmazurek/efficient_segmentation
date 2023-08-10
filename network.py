""" source: https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py"""

from collections import OrderedDict
import lightning.pytorch as pl
import torch
import torch.nn as nn
from utils import DiceLoss
from torchmetrics import Dice
import segmentation_models_pytorch as smp
