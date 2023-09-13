from torch.nn.functional import binary_cross_entropy_with_logits
from segmentation_models_pytorch.losses import DiceLoss, MCCLoss
from models.torch_models import (
    UNet,
    AttSqueezeUNet,
    MobileNetV3Seg,
    MicroNet,
    SegNet,
    SQNet,
    LinkNet,
    FSSNet,
    FPENet,
    ESPNet,
    ESNet,
    ERFNet,
    ENet,
    EDANet,
    DABNet,
    ContextNet,
    CGNet,
)
from models.ESPNet_v2.seg_model import EESPNet_Seg

AVAILABLE_MODELS = [
    "Unet",
    "AttSqueezeUnet",
    "MobileNetV3",
    "MicroNet",
    "SegNet",
    "SQNet",
    "LinkNet",
    "FSSNet",
    "FPENet",
    "ESPNet",
    "ESNet",
    "ERFNet",
    "ENet",
    "EDANet",
    "DABNet",
    "ContextNet",
    "CGNet",
    "ESPNetv2",
]
AVAILABLE_LOSSES = ["DiceLoss", "MCCLoss", "BCE"]


def return_chosen_model(model_name, in_channels, in_shape, init_features):
    if model_name == "Unet":
        return UNet(in_channels, in_channels, init_features)
    elif model_name == "AttSqueezeUnet":
        return AttSqueezeUNet(1, in_shape)
    elif model_name == "MobileNetV3":
        return MobileNetV3Seg(nclass=1, pretrained_base=False)
    elif model_name == "MicroNet":
        return MicroNet(nb_classes=1, inputs_shape=in_shape[1:])
    elif model_name == "SegNet":
        return SegNet(classes=in_channels)
    elif model_name == "SQNet":
        return SQNet(classes=in_channels)
    elif model_name == "LinkNet":
        return LinkNet(classes=in_channels)
    elif model_name == "FSSNet":
        return FSSNet(classes=in_channels)
    elif model_name == "FPENet":
        return FPENet(classes=in_channels)
    elif model_name == "ESPNet":
        return ESPNet(classes=in_channels)
    elif model_name == "ESNet":
        return ESNet(classes=in_channels)
    elif model_name == "ERFNet":
        return ERFNet(classes=in_channels)
    elif model_name == "ENet":
        return ENet(classes=in_channels)
    elif model_name == "EDANet":
        return EDANet(classes=in_channels)
    elif model_name == "DABNet":
        return DABNet(classes=in_channels)
    elif model_name == "ContextNet":
        return ContextNet(classes=in_channels)
    elif model_name == "CGNet":
        return CGNet(classes=in_channels)
    elif model_name == "ESPNetv2":
        return EESPNet_Seg(classes=in_channels)


def return_chosen_loss(loss_name):
    if loss_name == "DiceLoss":
        return DiceLoss(mode="binary", from_logits=False)
    elif loss_name == "MCCLoss":
        return MCCLoss()
    elif loss_name == "BCE":
        return binary_cross_entropy_with_logits
