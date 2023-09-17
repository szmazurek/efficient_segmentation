from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torchvision.models import resnet
from .activations import NON_LINEARITY

"""Some models come from
https://github.com/xiaoyufenfei/Efficient-Segmentation-Networks"""


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        init_features=64,
    ):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet._block(
            features * 8, features * 16, name="bottleneck"
        )

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block(
            (features * 8) * 2, features * 8, name="dec4"
        )
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block(
            (features * 4) * 2, features * 4, name="dec3"
        )
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block(
            (features * 2) * 2, features * 2, name="dec2"
        )
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))
        # return self.conv(dec1)

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )


class FireModule(nn.Module):
    def __init__(self, fire_id, squeeze, expand, in_channels=3):
        super(FireModule, self).__init__()

        self.fire = nn.Sequential(
            nn.Conv2d(in_channels, squeeze, kernel_size=1, padding="same"),
            nn.BatchNorm2d(squeeze),
            nn.ReLU(inplace=True),
        )
        self.left = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.right = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fire(x)
        left = self.left(x)
        right = self.right(x)
        x = torch.cat([left, right], dim=1)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(
                F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(
                F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(
                F_int, 1, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UpsamplingBlock(nn.Module):
    def __init__(
        self,
        filters,
        fire_id,
        squeeze,
        expand,
        strides,
        padding,
        deconv_ksize,
        att_filters,
        x_input_shape,
        g_input_shape,
    ):
        super(UpsamplingBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(
            in_channels=x_input_shape[1],
            out_channels=filters,
            kernel_size=deconv_ksize,
            stride=strides,
            padding=padding,
            output_padding=0 if strides == (1, 1) else 1,
        )
        x_dummy = torch.zeros(x_input_shape)
        g_dummy = torch.zeros(g_input_shape)
        x_dummy_shape = self.upconv(x_dummy).shape
        self.attention = AttentionBlock(
            F_g=x_dummy_shape[1], F_l=g_input_shape[1], F_int=att_filters
        )
        g_dummy_shape = self.upconv_attention_block(x_dummy, g_dummy).shape
        self.fire = FireModule(
            fire_id, squeeze, expand, in_channels=g_dummy_shape[1]
        )

    def upconv_attention_block(self, x, g):
        d = self.upconv(x)
        x = self.attention(d, g)
        d = torch.cat([x, d], axis=1)
        return d

    def forward(self, x, g):
        d = self.upconv_attention_block(x, g)
        x = self.fire(d)
        return x


class AttSqueezeUNet(nn.Module):
    def __init__(self, n_classes, in_shape, dropout=False):
        super(AttSqueezeUNet, self).__init__()
        self._dropout = dropout
        x1_shape = [int(x / 2) for x in in_shape[-2:]]
        x2_shape = [int(x / 4) for x in in_shape[-2:]]
        x3_shape = [int(x / 8) for x in in_shape[-2:]]
        x4_shape = [int(x / 16) for x in in_shape[-2:]]
        padding_1 = self.calculate_same_padding(
            in_shape[-2:],
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.conv_1 = nn.Conv2d(
            in_shape[1], 64, kernel_size=3, stride=(2, 2), padding=padding_1
        )
        padding_2 = self.calculate_same_padding(
            x1_shape,
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.max_pooling_1 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=padding_2
        )
        self.fire_1 = FireModule(1, 16, 64, in_channels=64)
        self.fire_2 = FireModule(2, 16, 64, in_channels=128)
        padding_3 = self.calculate_same_padding(
            x2_shape,
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.max_pooling_2 = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=padding_3
        )

        self.fire_3 = FireModule(3, 32, 128, in_channels=128)
        self.fire_4 = FireModule(4, 32, 128, in_channels=256)
        padding_4 = self.calculate_same_padding(
            x3_shape,
            kernel_size=(3, 3),
            stride=(2, 2),
            dilation=(1, 1),
        )
        self.max_pooling_3 = nn.MaxPool2d(
            kernel_size=(3, 3), stride=(2, 2), padding=padding_4
        )
        self.fire_5 = FireModule(5, 48, 192, in_channels=256)
        self.fire_6 = FireModule(6, 48, 192, in_channels=384)
        self.fire_7 = FireModule(7, 64, 256, in_channels=384)
        self.fire_8 = FireModule(8, 64, 256, in_channels=512)
        padding_5 = self.calculate_same_padding(
            x4_shape,
            kernel_size=(3, 3),
            stride=(1, 1),
            dilation=(1, 1),
        )
        self.upsampling_1 = UpsamplingBlock(
            filters=192,
            fire_id=9,
            squeeze=48,
            expand=192,
            padding=padding_5,
            strides=(1, 1),
            deconv_ksize=3,
            att_filters=96,
            x_input_shape=(1, 512, x4_shape[0], x4_shape[1]),
            g_input_shape=(1, 384, x4_shape[0], x4_shape[1]),
        )
        padding_6 = padding_5
        self.upsampling_2 = UpsamplingBlock(
            filters=128,
            fire_id=10,
            squeeze=32,
            expand=128,
            strides=(1, 1),
            deconv_ksize=3,
            att_filters=64,
            padding=padding_6,
            x_input_shape=(1, 384, x4_shape[0], x4_shape[1]),
            g_input_shape=(1, 256, x4_shape[0], x4_shape[1]),
        )
        padding_7 = self.calculate_same_padding(
            x4_shape, kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)
        )
        self.upsampling_3 = UpsamplingBlock(
            filters=64,
            fire_id=11,
            squeeze=16,
            expand=64,
            padding=padding_7,
            strides=(2, 2),
            deconv_ksize=3,
            att_filters=16,
            x_input_shape=(1, 256, x4_shape[0], x4_shape[1]),
            g_input_shape=(1, 128, x3_shape[0], x3_shape[1]),
        )
        padding_8 = self.calculate_same_padding(
            x3_shape, kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1)
        )
        self.upsampling_4 = UpsamplingBlock(
            filters=32,
            fire_id=12,
            squeeze=16,
            expand=32,
            padding=padding_8,
            strides=(2, 2),
            deconv_ksize=3,
            att_filters=4,
            x_input_shape=(1, 128, x3_shape[0], x3_shape[1]),
            g_input_shape=(1, 64, x2_shape[0], x2_shape[1]),
        )

        self.upsampling_5 = nn.Upsample(scale_factor=2)
        self.upsampling_6 = nn.Upsample(scale_factor=2)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, padding="same"),
            nn.Softmax(dim=1) if n_classes > 1 else nn.Sigmoid(),
        )

    @staticmethod
    def calculate_same_padding(input_size, kernel_size, stride, dilation):
        if input_size[0] % stride[0] == 0:
            pad_along_height = max(
                dilation[0] * (kernel_size[0] - stride[0]), 0
            )
        else:
            pad_along_height = max(
                dilation[0] * (kernel_size[0] - (input_size[0] % stride[0])), 0
            )

        if input_size[1] % stride[1] == 0:
            pad_along_width = max(
                dilation[1] * (kernel_size[1] - stride[1]), 0
            )
        else:
            pad_along_width = max(
                dilation[1] * (kernel_size[1] - (input_size[1] % stride[1])), 0
            )

        p1 = math.ceil(pad_along_height / 2)
        p2 = math.ceil(pad_along_width / 2)
        return (p1, p2)

    @staticmethod
    def calculate_padding_equal(
        kernel_size, stride, input_size, output_size=None, dilation=(1, 1)
    ):
        """Calculate padding to keep the same input and output size."""

        if output_size is None:
            output_size = input_size  # assume same padding
        p1 = math.ceil(
            (
                (output_size[0] - 1) * stride[0]
                + 1
                + dilation[0] * (kernel_size[0] - 1)
                - input_size[0]
            )
            / 2
        )
        p2 = math.ceil(
            (
                (output_size[1] - 1) * stride[1]
                + 1
                + dilation[1] * (kernel_size[1] - 1)
                - input_size[1]
            )
            / 2
        )
        return (p1, p2)

    def forward(self, x):
        x0 = self.conv_1(x)
        x1 = self.max_pooling_1(x0)

        x2 = self.fire_1(x1)
        x2 = self.fire_2(x2)
        x2 = self.max_pooling_2(x2)

        x3 = self.fire_3(x2)
        x3 = self.fire_4(x3)
        x3 = self.max_pooling_3(x3)

        x4 = self.fire_5(x3)
        x4 = self.fire_6(x4)

        x5 = self.fire_7(x4)
        x5 = self.fire_8(x5)

        if self._dropout:
            x5 = F.dropout(x5, p=0.5)
        d5 = self.upsampling_1(x5, x4)
        d4 = self.upsampling_2(d5, x3)
        d3 = self.upsampling_3(d4, x2)
        d2 = self.upsampling_4(d3, x1)
        d1 = self.upsampling_5(d2)

        d0 = torch.cat([d1, x0], axis=1)
        d0 = self.conv_2(d0)
        d0 = self.upsampling_6(d0)
        d = self.conv_3(d0)

        return d


"""Searching for MobileNetV3"""
__all__ = ["MobileNetV3", "get_mobilenet_v3", "mobilenet_v3_small_1_0"]

__all__ = ["Hswish", "ConvBNHswish", "Bottleneck", "SEModule"]


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.0) / 6.0


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu6(x + 3.0) / 6.0


class ConvBNHswish(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm_layer=nn.BatchNorm2d,
        **kwargs
    ):
        super(ConvBNHswish, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=False,
        )
        self.bn = norm_layer(out_channels)
        self.act = Hswish(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEModuleEfficientNetv3(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModuleEfficientNetv3, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            Hsigmoid(True),
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        out = self.avg_pool(x).view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        return x * out.expand_as(x)


class Identity(nn.Module):
    def __init__(self, in_channels):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        exp_size,
        kernel_size,
        stride,
        dilation=1,
        se=False,
        nl="RE",
        norm_layer=nn.BatchNorm2d,
        **kwargs
    ):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if nl == "HS":
            act = Hswish
        else:
            act = nn.ReLU
        if se:
            SELayer = SEModuleEfficientNetv3
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(in_channels, exp_size, 1, bias=False),
            norm_layer(exp_size),
            act(True),
            # dw
            nn.Conv2d(
                exp_size,
                exp_size,
                kernel_size,
                stride,
                (kernel_size - 1) // 2 * dilation,
                dilation,
                groups=exp_size,
                bias=False,
            ),
            norm_layer(exp_size),
            SELayer(exp_size),
            act(True),
            # pw-linear
            nn.Conv2d(exp_size, out_channels, 1, bias=False),
            norm_layer(out_channels),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


# if __name__ == '__main__':
#     img = torch.randn(1, 16, 64, 64)
#     model = Bottleneck(16, 16, 16, 3, 1)
#     out = model(img)
#     print(out.size())


class MobileNetV3(nn.Module):
    def __init__(
        self,
        nclass=2,
        mode="small",
        width_mult=1.0,
        dilated=True,
        pretrained_base=True,
        norm_layer=nn.BatchNorm2d,
    ):
        super(MobileNetV3, self).__init__()
        if mode == "large":
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, False, "RE", 1],
                [3, 64, 24, False, "RE", 2],
                [3, 72, 24, False, "RE", 1],
            ]
            layer2_setting = [
                [5, 72, 40, True, "RE", 2],
                [5, 120, 40, True, "RE", 1],
                [5, 120, 40, True, "RE", 1],
            ]
            layer3_setting = [
                [3, 240, 80, False, "HS", 2],
                [3, 200, 80, False, "HS", 1],
                [3, 184, 80, False, "HS", 1],
                [3, 184, 80, False, "HS", 1],
                [3, 480, 112, True, "HS", 1],
                [3, 672, 112, True, "HS", 1],
                [5, 672, 112, True, "HS", 1],
            ]
            layer4_setting = [
                [5, 672, 160, True, "HS", 2],
                [5, 960, 160, True, "HS", 1],
            ]
        elif mode == "small":
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, True, "RE", 2],
            ]
            layer2_setting = [
                [3, 72, 24, False, "RE", 2],
                [3, 88, 24, False, "RE", 1],
            ]
            layer3_setting = [
                [5, 96, 40, True, "HS", 2],
                [5, 240, 40, True, "HS", 1],
                [5, 240, 40, True, "HS", 1],
                [5, 120, 48, True, "HS", 1],
                [5, 144, 48, True, "HS", 1],
            ]
            layer4_setting = [
                [5, 288, 96, True, "HS", 2],
                [5, 576, 96, True, "HS", 1],
                [5, 576, 96, True, "HS", 1],
            ]
        else:
            raise ValueError("Unknown mode.")

        # building first layer
        input_channels = int(16 * width_mult) if width_mult > 1.0 else 16
        self.conv1 = ConvBNHswish(
            1, input_channels, 3, 2, 1, norm_layer=norm_layer
        )

        # building bottleneck blocks
        self.layer1, input_channels = self._make_layer(
            Bottleneck,
            input_channels,
            layer1_setting,
            width_mult,
            norm_layer=norm_layer,
        )
        self.layer2, input_channels = self._make_layer(
            Bottleneck,
            input_channels,
            layer2_setting,
            width_mult,
            norm_layer=norm_layer,
        )
        self.layer3, input_channels = self._make_layer(
            Bottleneck,
            input_channels,
            layer3_setting,
            width_mult,
            norm_layer=norm_layer,
        )
        if dilated:
            self.layer4, input_channels = self._make_layer(
                Bottleneck,
                input_channels,
                layer4_setting,
                width_mult,
                dilation=2,
                norm_layer=norm_layer,
            )
        else:
            self.layer4, input_channels = self._make_layer(
                Bottleneck,
                input_channels,
                layer4_setting,
                width_mult,
                norm_layer=norm_layer,
            )

        # building last several layers
        classifier = list()
        if mode == "large":
            last_bneck_channels = (
                int(960 * width_mult) if width_mult > 1.0 else 960
            )
            self.conv5 = ConvBNHswish(
                input_channels, last_bneck_channels, 1, norm_layer=norm_layer
            )
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        elif mode == "small":
            last_bneck_channels = (
                int(576 * width_mult) if width_mult > 1.0 else 576
            )
            self.conv5 = ConvBNHswish(
                input_channels, last_bneck_channels, 1, norm_layer=norm_layer
            )
            classifier.append(SEModule(last_bneck_channels))
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        else:
            raise ValueError("Unknown mode.")
        self.classifier = nn.Sequential(*classifier)

        self._init_weights()

    def _make_layer(
        self,
        block,
        input_channels,
        block_setting,
        width_mult,
        dilation=1,
        norm_layer=nn.BatchNorm2d,
    ):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            s = 1 if dilation != 1 else s
            out_channels = int(c * width_mult)
            exp_channels = int(exp_size * width_mult)
            layers.append(
                block(
                    input_channels,
                    out_channels,
                    exp_channels,
                    k,
                    s,
                    dilation,
                    se,
                    nl,
                    norm_layer,
                )
            )
            input_channels = out_channels
        return nn.Sequential(*layers), input_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        x = self.classifier(x)
        x = x.view(x.size(0), x.size(1))
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# def get_mobilenet_v3(mode='small', width_mult=1.0, pretrained=False, root='~/,torch/models', **kwargs):
def get_mobilenet_v3(width_mult=1.0, dilated=True, pretrained=False, **kwargs):
    model = MobileNetV3(
        mode="small", width_mult=width_mult, pretrained_base=True, **kwargs
    )
    if pretrained:
        raise ValueError("Not support pretrained")
    return model


# def mobilenet_v3_small_1_0(**kwargs):
#     return get_mobilenet_v3(width_mult=1.0, **kwargs)a


def mobilenet_v3_small_1_0(width_mult=1.0, **kwargs):
    return get_mobilenet_v3(width_mult=width_mult, **kwargs)


# if __name__ == '__main__':
#     model = mobilenet_v3_small_1_0()


# Base Model for Semantic Segmentation
class SegBaseModel(nn.Module):
    def __init__(
        self,
        nclass,
        aux=False,
        backbone="mobilenetv3_small",
        pretrained_base=True,
        **kwargs
    ):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.mode = backbone.split("_")[-1]
        assert self.mode in ["large", "small"]
        if backbone == "mobilenetv3_small":
            self.pretrained = mobilenet_v3_small_1_0(
                dilated=True, pretrained=pretrained_base, **kwargs
            )
        else:
            raise RuntimeError("unknown backbone: {}".format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)

        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        c4 = self.pretrained.conv5(c4)

        return c1, c2, c3, c4


# MobileNetV3 for Semantic Segmentation
class MobileNetV3Seg(SegBaseModel):
    def __init__(
        self,
        nclass,
        aux=False,
        backbone="mobilenetv3_small",
        pretrained_base=True,
        **kwargs
    ):
        super(MobileNetV3Seg, self).__init__(
            nclass, aux, backbone, pretrained_base, **kwargs
        )
        self.head = _SegHead(nclass, self.mode, **kwargs)
        if aux:
            inter_channels = 40 if self.mode == "small" else 24
            self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c2)
            auxout = F.interpolate(
                auxout, size, mode="bilinear", align_corners=True
            )
            outputs.append(auxout)
        return F.sigmoid(x)  # tuple(outputs)


class _SegHead(nn.Module):
    def __init__(
        self, nclass, mode="small", norm_layer=nn.BatchNorm2d, **kwargs
    ):
        super(_SegHead, self).__init__()
        in_channels = 960 if mode == "large" else 576
        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)


class _LRASPP(nn.Module):
    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
        )
        self.b1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Use adaptive average pooling
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode="bilinear", align_corners=True)
        x = feat1 * feat2
        return x


class FireModuleMicronet(nn.Module):
    def __init__(
        self,
        fire_i,
        base_e,
        freq,
        squeeze_ratio,
        pct_3x3,
        dilation_rate,
        activation,
        kernel_initializer,
        data_format,
        use_bias=False,
        decoder=False,
    ):
        super(FireModuleMicronet, self).__init__()
        e_i, s_1x1, e_1x1, e_3x3 = self.get_fire_config(
            fire_i, base_e, freq, squeeze_ratio, pct_3x3
        )
        self.decoder = decoder

        if decoder:
            d = "decoder_"
        else:
            d = ""

        self.squeeze = nn.Conv2d(
            in_channels=e_i, out_channels=s_1x1, kernel_size=1, bias=use_bias
        )
        self.fire2_expand1 = nn.Conv2d(
            in_channels=s_1x1,
            out_channels=e_1x1,
            kernel_size=1,
            bias=use_bias,
            padding="same",
        )
        self.fire2_expand2 = nn.Conv2d(
            in_channels=s_1x1,
            out_channels=e_3x3,
            kernel_size=3,
            padding="same",
            dilation=dilation_rate,
            bias=use_bias,
        )

    def forward(self, inputs):
        squeeze = self.squeeze(inputs)
        fire2_expand1 = self.fire2_expand1(F.relu(squeeze))
        fire2_expand2 = self.fire2_expand2(F.relu(squeeze))
        merge = torch.cat([fire2_expand1, fire2_expand2], dim=1)
        return merge

    def get_fire_config(self, i, base_e, freq, squeeze_ratio, pct_3x3):
        e_i = base_e * (2 ** (i // freq))
        s_1x1 = int(squeeze_ratio * e_i)
        e_3x3 = int(pct_3x3 * e_i)
        e_1x1 = e_i - e_3x3
        return e_i, s_1x1, e_1x1, e_3x3


class MicroNet(nn.Module):
    def __init__(
        self,
        nb_classes=2,
        base_e=64,
        freq=4,
        squeeze_ratio=0.25,
        pct_3x3=0.5,
        inputs_shape=(1, 224, 224),
        use_bias=False,
        data_format="channels_first",
        activation=nn.ReLU(),
        kernel_initializer=None,
    ):
        super(MicroNet, self).__init__()

        self.inputs = nn.Conv2d(inputs_shape[0], base_e, kernel_size=1)

        # Encoder
        self.conv1 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv2 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv3 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            2,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv4 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            3,
            activation,
            kernel_initializer,
            data_format,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv6 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv7 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            2,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv8 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            3,
            activation,
            kernel_initializer,
            data_format,
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv10 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv11 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            2,
            activation,
            kernel_initializer,
            data_format,
        )
        self.conv12 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            3,
            activation,
            kernel_initializer,
            data_format,
        )

        # Decoder
        self.d_conv11 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            3,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.d_conv10 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            2,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.d_conv9 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.up2 = nn.ConvTranspose2d(
            in_channels=base_e, out_channels=base_e, kernel_size=2, stride=2
        )

        self.d_conv8 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            3,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.d_conv7 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            2,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.d_conv6 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.up1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=2
        )

        self.d_conv5 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            3,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.d_conv4 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            2,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )
        self.d_conv3 = self.create_fire_modules(
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            1,
            activation,
            kernel_initializer,
            data_format,
            decoder=True,
        )

        # Classifier
        self.out_conv = nn.Conv2d(base_e, nb_classes, kernel_size=1)

    def forward(self, x):
        inputs = self.inputs(x)

        # Encoder
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        pool1 = self.pool1(conv4)

        conv5 = self.conv5(pool1)
        conv6 = self.conv6(conv5)
        conv7 = self.conv7(conv6)
        conv8 = self.conv8(conv7)
        pool2 = self.pool2(conv8)

        conv9 = self.conv9(pool2)
        conv10 = self.conv10(conv9)
        conv11 = self.conv11(conv10)
        conv12 = self.conv12(conv11)

        # Decoder
        d_conv11 = self.d_conv11(conv12)
        d_conv10 = self.d_conv10(d_conv11)
        d_conv9 = self.d_conv9(d_conv10)
        up2 = self.up2(d_conv9)

        d_conv8 = self.d_conv8(up2 + conv8)
        d_conv7 = self.d_conv7(d_conv8)
        d_conv6 = self.d_conv6(d_conv7)
        up1 = self.up1(d_conv6)

        d_conv5 = self.d_conv5(up1 + conv4)
        d_conv4 = self.d_conv4(d_conv5)
        d_conv3 = self.d_conv3(d_conv4)

        # Classifier
        out_conv = self.out_conv(d_conv3)
        return F.sigmoid(out_conv)

    def create_fire_modules(
        self,
        base_e,
        freq,
        squeeze_ratio,
        pct_3x3,
        dilation_rate,
        activation,
        kernel_initializer,
        data_format,
        use_bias=False,
        decoder=False,
    ):
        return FireModuleMicronet(
            0,
            base_e,
            freq,
            squeeze_ratio,
            pct_3x3,
            dilation_rate,
            activation,
            kernel_initializer,
            data_format,
            use_bias,
            decoder,
        )


class SegNet(nn.Module):
    def __init__(self, classes=19):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = nn.Conv2d(64, classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)))
        x12 = F.relu(self.bn12(self.conv12(x11)))
        x1_size = x12.size()
        x1p, id1 = F.max_pool2d(
            x12, kernel_size=2, stride=2, return_indices=True
        )

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)))
        x22 = F.relu(self.bn22(self.conv22(x21)))
        x2_size = x22.size()
        x2p, id2 = F.max_pool2d(
            x22, kernel_size=2, stride=2, return_indices=True
        )

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)))
        x32 = F.relu(self.bn32(self.conv32(x31)))
        x33 = F.relu(self.bn33(self.conv33(x32)))
        x3_size = x33.size()
        x3p, id3 = F.max_pool2d(
            x33, kernel_size=2, stride=2, return_indices=True
        )

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)))
        x42 = F.relu(self.bn42(self.conv42(x41)))
        x43 = F.relu(self.bn43(self.conv43(x42)))
        x4_size = x43.size()
        x4p, id4 = F.max_pool2d(
            x43, kernel_size=2, stride=2, return_indices=True
        )

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)))
        x52 = F.relu(self.bn52(self.conv52(x51)))
        x53 = F.relu(self.bn53(self.conv53(x52)))
        x5_size = x53.size()
        x5p, id5 = F.max_pool2d(
            x53, kernel_size=2, stride=2, return_indices=True
        )

        # Stage 5d
        x5d = F.max_unpool2d(
            x5p, id5, kernel_size=2, stride=2, output_size=x5_size
        )
        x53d = F.relu(self.bn53d(self.conv53d(x5d)))
        x52d = F.relu(self.bn52d(self.conv52d(x53d)))
        x51d = F.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = F.max_unpool2d(
            x51d, id4, kernel_size=2, stride=2, output_size=x4_size
        )
        x43d = F.relu(self.bn43d(self.conv43d(x4d)))
        x42d = F.relu(self.bn42d(self.conv42d(x43d)))
        x41d = F.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = F.max_unpool2d(
            x41d, id3, kernel_size=2, stride=2, output_size=x3_size
        )
        x33d = F.relu(self.bn33d(self.conv33d(x3d)))
        x32d = F.relu(self.bn32d(self.conv32d(x33d)))
        x31d = F.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = F.max_unpool2d(
            x31d, id2, kernel_size=2, stride=2, output_size=x2_size
        )
        x22d = F.relu(self.bn22d(self.conv22d(x2d)))
        x21d = F.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = F.max_unpool2d(
            x21d, id1, kernel_size=2, stride=2, output_size=x1_size
        )
        x12d = F.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return F.sigmoid(x11d)


class Fire(nn.Module):
    def __init__(self, inplanes, squeeze_planes, expand_planes):
        super(Fire, self).__init__()
        self.conv1 = nn.Conv2d(
            inplanes, squeeze_planes, kernel_size=1, stride=1
        )
        # self.bn1 = nn.BatchNorm2d(squeeze_planes)
        self.relu1 = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(
            squeeze_planes, expand_planes, kernel_size=1, stride=1
        )
        # self.bn2 = nn.BatchNorm2d(expand_planes)
        self.conv3 = nn.Conv2d(
            squeeze_planes, expand_planes, kernel_size=3, stride=1, padding=1
        )
        # self.bn3 = nn.BatchNorm2d(expand_planes)
        self.relu2 = nn.ELU(inplace=True)

        # using MSR initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu1(x)
        out1 = self.conv2(x)
        # out1 = self.bn2(out1)
        out2 = self.conv3(x)
        # out2 = self.bn3(out2)
        out = torch.cat([out1, out2], 1)
        out = self.relu2(out)
        return out


class ParallelDilatedConv(nn.Module):
    def __init__(self, inplanes, planes):
        super(ParallelDilatedConv, self).__init__()
        self.dilated_conv_1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.dilated_conv_2 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=2, dilation=2
        )
        self.dilated_conv_3 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=3, dilation=3
        )
        self.dilated_conv_4 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=1, padding=4, dilation=4
        )
        self.relu1 = nn.ELU(inplace=True)
        self.relu2 = nn.ELU(inplace=True)
        self.relu3 = nn.ELU(inplace=True)
        self.relu4 = nn.ELU(inplace=True)

    def forward(self, x):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        out3 = self.dilated_conv_3(x)
        out4 = self.dilated_conv_4(x)
        out1 = self.relu1(out1)
        out2 = self.relu2(out2)
        out3 = self.relu3(out3)
        out4 = self.relu4(out4)
        out = out1 + out2 + out3 + out4
        return out


class SQNet(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.num_classes = classes

        self.conv1 = nn.Conv2d(1, 96, kernel_size=3, stride=2, padding=1)  # 32
        # self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ELU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16
        self.fire1_1 = Fire(96, 16, 64)
        self.fire1_2 = Fire(128, 16, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 8
        self.fire2_1 = Fire(128, 32, 128)
        self.fire2_2 = Fire(256, 32, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 4
        self.fire3_1 = Fire(256, 64, 256)
        self.fire3_2 = Fire(512, 64, 256)
        self.fire3_3 = Fire(512, 64, 256)
        self.parallel = ParallelDilatedConv(512, 512)
        self.deconv1 = nn.ConvTranspose2d(
            512, 256, 3, stride=2, padding=1, output_padding=1
        )
        # self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ELU(inplace=True)
        self.deconv2 = nn.ConvTranspose2d(
            512, 128, 3, stride=2, padding=1, output_padding=1
        )
        # self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ELU(inplace=True)
        self.deconv3 = nn.ConvTranspose2d(
            256, 96, 3, stride=2, padding=1, output_padding=1
        )
        # self.bn4 = nn.BatchNorm2d(96)
        self.relu4 = nn.ELU(inplace=True)
        self.deconv4 = nn.ConvTranspose2d(
            192, self.num_classes, 3, stride=2, padding=1, output_padding=1
        )

        self.conv3_1 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1
        )  # 32
        self.conv3_2 = nn.Conv2d(
            512, 512, kernel_size=3, stride=1, padding=1
        )  # 32
        self.conv2_1 = nn.Conv2d(
            128, 128, kernel_size=3, stride=1, padding=1
        )  # 32
        self.conv2_2 = nn.Conv2d(
            256, 256, kernel_size=3, stride=1, padding=1
        )  # 32
        self.conv1_1 = nn.Conv2d(
            96, 96, kernel_size=3, stride=1, padding=1
        )  # 32
        self.conv1_2 = nn.Conv2d(
            192, 192, kernel_size=3, stride=1, padding=1
        )  # 32

        self.relu1_1 = nn.ELU(inplace=True)
        self.relu1_2 = nn.ELU(inplace=True)
        self.relu2_1 = nn.ELU(inplace=True)
        self.relu2_2 = nn.ELU(inplace=True)
        self.relu3_1 = nn.ELU(inplace=True)
        self.relu3_2 = nn.ELU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        x_1 = self.relu1(x)
        # print "x_1: %s" % str(x_1.size())
        x = self.maxpool1(x_1)
        x = self.fire1_1(x)
        x_2 = self.fire1_2(x)
        # print "x_2: %s" % str(x_2.size())
        x = self.maxpool2(x_2)
        x = self.fire2_1(x)
        x_3 = self.fire2_2(x)
        # print "x_3: %s" % str(x_3.size())
        x = self.maxpool3(x_3)
        x = self.fire3_1(x)
        x = self.fire3_2(x)
        x = self.fire3_3(x)
        x = self.parallel(x)
        # print "x: %s" % str(x.size())
        y_3 = self.deconv1(x)
        y_3 = self.relu2(y_3)
        x_3 = self.conv3_1(x_3)
        x_3 = self.relu3_1(x_3)
        # print "y_3: %s" % str(y_3.size())
        # x = x.transpose(1, 2, 0)
        # print('x_3.size():', x_3.size())
        # print('y_3.size():', y_3.size())
        x_3 = F.interpolate(
            x_3, y_3.size()[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x_3, y_3], 1)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        # concat x_3
        y_2 = self.deconv2(x)
        y_2 = self.relu3(y_2)
        x_2 = self.conv2_1(x_2)
        x_2 = self.relu2_1(x_2)
        # print "y_2: %s" % str(y_2.size())
        # concat x_2
        # print('x_2.size():', x_2.size())
        # print('y_2.size():', y_2.size())
        y_2 = F.interpolate(
            y_2, x_2.size()[2:], mode="bilinear", align_corners=True
        )
        x = torch.cat([x_2, y_2], 1)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        y_1 = self.deconv3(x)
        y_1 = self.relu4(y_1)
        x_1 = self.conv1_1(x_1)
        x_1 = self.relu1_1(x_1)
        # print "y_1: %s" % str(y_1.size())
        # concat x_1
        x = torch.cat([x_1, y_1], 1)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.deconv4(x)
        return F.sigmoid(x)  # , x_1, x_2, x_3, y_1, y_2, y_3


class BasicBlockLinkNet(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super(BasicBlockLinkNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_planes,
            out_planes,
            kernel_size,
            1,
            padding,
            groups=groups,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.relu(out + residual)

        return out


class LinkNetEncoder(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        groups=1,
        bias=False,
    ):
        super(LinkNetEncoder, self).__init__()
        self.block1 = BasicBlockLinkNet(
            in_planes, out_planes, kernel_size, stride, padding, groups, bias
        )
        self.block2 = BasicBlockLinkNet(
            out_planes, out_planes, kernel_size, 1, padding, groups, bias
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)

        return x


class LinkNetDecoder(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=False,
    ):
        # TODO bias=True
        super(LinkNetDecoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // 4, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
        )
        self.tp_conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_planes // 4,
                in_planes // 4,
                kernel_size,
                stride,
                padding,
                output_padding,
                bias=bias,
            ),
            nn.BatchNorm2d(in_planes // 4),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_planes // 4, out_planes, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_high_level, x_low_level):
        x = self.conv1(x_high_level)
        x = self.tp_conv(x)

        # solution for padding issues
        # diffY = x_low_level.size()[2] - x_high_level.size()[2]
        # diffX = x_low_level.size()[3] - x_high_level.size()[3]
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = center_crop(x, x_low_level.size()[2], x_low_level.size()[3])

        x = self.conv2(x)

        return x


def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    diffy = (h - max_height) // 2
    diffx = (w - max_width) // 2
    return layer[
        :, :, diffy : (diffy + max_height), diffx : (diffx + max_width)
    ]


def up_pad(layer, skip_height, skip_width):
    _, _, h, w = layer.size()
    diffy = skip_height - h
    diffx = skip_width - w
    return F.pad(
        layer, [diffx // 2, diffx - diffx // 2, diffy // 2, diffy - diffy // 2]
    )


class LinkNetImprove(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()

        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = LinkNetDecoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = LinkNetDecoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = LinkNetDecoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = LinkNetDecoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.in_block(x)

        # LinkNetEncoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # LinkNetDecoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return y


class LinkNet(nn.Module):
    """
    Generate model architecture
    """

    def __init__(self, classes=19):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.encoder1 = LinkNetEncoder(64, 64, 3, 1, 1)
        self.encoder2 = LinkNetEncoder(64, 128, 3, 2, 1)
        self.encoder3 = LinkNetEncoder(128, 256, 3, 2, 1)
        self.encoder4 = LinkNetEncoder(256, 512, 3, 2, 1)

        self.decoder4 = LinkNetDecoder(512, 256, 3, 2, 1, 1)
        self.decoder3 = LinkNetDecoder(256, 128, 3, 2, 1, 1)
        self.decoder2 = LinkNetDecoder(128, 64, 3, 2, 1, 1)
        self.decoder1 = LinkNetDecoder(64, 64, 3, 1, 1, 0)

        # Classifier
        self.tp_conv1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.tp_conv2 = nn.ConvTranspose2d(32, classes, 2, 2, 0)

    def forward(self, x):
        # Initial block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # LinkNetEncoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # LinkNetDecoder blocks
        d4 = e3 + self.decoder4(e4, e3)
        d3 = e2 + self.decoder3(d4, e2)
        d2 = e1 + self.decoder2(d3, e1)
        d1 = x + self.decoder1(d2, x)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)

        return F.sigmoid(y)


class InitialBlock(nn.Module):
    def __init__(self, ninput, noutput, non_linear="ReLU"):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=False
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput - ninput, eps=1e-3)
        self.relu = NON_LINEARITY[non_linear]

    def forward(self, input):
        output = self.relu(self.bn(self.conv(input)))
        output = torch.cat([output, self.pool(input)], 1)

        return output


class DownsamplingBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dropout_prob=0.0,
        bias=False,
        non_linear="ReLU",
    ):
        super().__init__()
        # Store parameters that are needed later
        internal_channels = in_channels // internal_ratio

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=bias
            ),
        )

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2, no padding
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear],
        )
        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear],
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            NON_LINEARITY[non_linear],
        )
        self.ext_regul = nn.Dropout2d(p=dropout_prob)
        # PReLU layer to apply after concatenating the branches
        self.out_prelu = NON_LINEARITY[non_linear]

    def forward(self, x):
        # Main branch shortcut
        main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = self.out_prelu(main + ext)

        return out


class UpsamplingBottleneck(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        kernel_size=2,
        padding=0,
        dropout_prob=0.0,
        bias=False,
        non_linear="ReLU",
    ):
        super().__init__()
        internal_channels = in_channels // internal_ratio

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        # self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias
            ),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear],
        )
        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=0,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            NON_LINEARITY[non_linear],
        )
        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            NON_LINEARITY[non_linear],
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = NON_LINEARITY[non_linear]

    def forward(self, x, x_pre):
        # Main branch shortcut         # here different origin paper, Fig 4 contradict to Fig 9
        main = x + x_pre

        main = self.main_conv1(main)  # 2. conv first, follow up

        main = F.interpolate(
            main, scale_factor=2, mode="bilinear", align_corners=True
        )  # 1. up first, follow conv
        # main = self.main_conv1(main)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = self.out_prelu(main + ext)

        return out


class DilatedBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        dropout_prob=0.0,
        bias=False,
        non_linear="ReLU",
    ):
        super(DilatedBlock, self).__init__()
        self.relu = NON_LINEARITY[non_linear]
        self.internal_channels = in_channels // 4
        # compress conv
        self.conv1 = nn.Conv2d(
            in_channels, self.internal_channels, 1, bias=bias
        )
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv2 = nn.Conv2d(
            self.internal_channels,
            self.internal_channels,
            kernel_size,
            stride,
            padding=int((kernel_size - 1) / 2 * dilation),
            dilation=dilation,
            groups=1,
            bias=bias,
        )
        self.conv2_bn = nn.BatchNorm2d(self.internal_channels)
        # a relu
        self.conv4 = nn.Conv2d(
            self.internal_channels, out_channels, 1, bias=bias
        )
        self.conv4_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        main = self.relu(self.conv1_bn(self.conv1(x)))
        main = self.relu(self.conv2_bn(self.conv2(main)))
        main = self.conv4_bn(self.conv4(main))
        main = self.regul(main)
        out = self.relu(torch.add(main, residual))
        return out


class Factorized_Block(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        dilation=1,
        dropout_prob=0.0,
        bias=False,
        non_linear="ReLU",
    ):
        super(Factorized_Block, self).__init__()
        self.relu = NON_LINEARITY[non_linear]
        self.internal_channels = in_channels // 4
        self.compress_conv1 = nn.Conv2d(
            in_channels, self.internal_channels, 1, padding=0, bias=bias
        )
        self.conv1_bn = nn.BatchNorm2d(self.internal_channels)
        # here is relu
        self.conv2_1 = nn.Conv2d(
            self.internal_channels,
            self.internal_channels,
            (kernel_size, 1),
            stride=(stride, 1),
            padding=(int((kernel_size - 1) / 2 * dilation), 0),
            dilation=(dilation, 1),
            bias=bias,
        )
        self.conv2_1_bn = nn.BatchNorm2d(self.internal_channels)
        self.conv2_2 = nn.Conv2d(
            self.internal_channels,
            self.internal_channels,
            (1, kernel_size),
            stride=(1, stride),
            padding=(0, int((kernel_size - 1) / 2 * dilation)),
            dilation=(1, dilation),
            bias=bias,
        )
        self.conv2_2_bn = nn.BatchNorm2d(self.internal_channels)
        # here is relu
        self.extend_conv3 = nn.Conv2d(
            self.internal_channels, out_channels, 1, padding=0, bias=bias
        )

        self.conv3_bn = nn.BatchNorm2d(out_channels)
        self.regul = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        residual = x
        main = self.relu((self.conv1_bn(self.compress_conv1(x))))
        main = self.relu(self.conv2_1_bn(self.conv2_1(main)))
        main = self.relu(self.conv2_2_bn(self.conv2_2(main)))

        main = self.conv3_bn(self.extend_conv3(main))
        main = self.regul(main)
        out = self.relu((torch.add(residual, main)))
        return out


class FSSNet(nn.Module):
    def __init__(self, classes):
        super().__init__()

        self.initial_block = InitialBlock(1, 16)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16, 64, padding=1, dropout_prob=0.03
        )
        self.factorized1_1 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_2 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_3 = Factorized_Block(64, 64, dropout_prob=0.03)
        self.factorized1_4 = Factorized_Block(64, 64, dropout_prob=0.03)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64, 128, padding=1, dropout_prob=0.3
        )
        self.dilated2_1 = DilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_2 = DilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_3 = DilatedBlock(128, 128, dilation=9, dropout_prob=0.3)
        self.dilated2_4 = DilatedBlock(128, 128, dilation=2, dropout_prob=0.3)
        self.dilated2_5 = DilatedBlock(128, 128, dilation=5, dropout_prob=0.3)
        self.dilated2_6 = DilatedBlock(128, 128, dilation=9, dropout_prob=0.3)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.3)
        self.bottleneck4_1 = DilatedBlock(64, 64, dropout_prob=0.3)
        self.bottleneck4_2 = DilatedBlock(64, 64, dropout_prob=0.3)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.3)
        self.bottleneck5_1 = DilatedBlock(16, 16, dropout_prob=0.3)
        self.bottleneck5_2 = DilatedBlock(16, 16, dropout_prob=0.3)

        self.transposed_conv = nn.ConvTranspose2d(
            16,
            classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

    def forward(self, x):
        # Initial block
        # Initial block
        x = self.initial_block(x)

        # Encoder - Block 1
        x_1 = self.downsample1_0(x)
        x = self.factorized1_1(x_1)
        x = self.factorized1_2(x)
        x = self.factorized1_3(x)
        x = self.factorized1_4(x)

        # Encoder - Block 2
        x_2 = self.downsample2_0(x)
        # print(x_2.shape)
        x = self.dilated2_1(x_2)
        x = self.dilated2_2(x)
        x = self.dilated2_3(x)
        x = self.dilated2_4(x)
        x = self.dilated2_5(x)
        x = self.dilated2_6(x)
        # print(x.shape)

        # Decoder - Block 3
        x = self.upsample4_0(x, x_2)
        x = self.bottleneck4_1(x)
        x = self.bottleneck4_2(x)

        # Decoder - Block 4
        x = self.upsample5_0(x, x_1)
        x = self.bottleneck5_1(x)
        x = self.bottleneck5_2(x)

        # Fullconv - DeConv
        x = self.transposed_conv(x)

        return F.sigmoid(x)


def conv3x3(
    in_planes,
    out_planes,
    stride=1,
    padding=1,
    dilation=1,
    groups=1,
    bias=False,
):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=1, stride=stride, bias=bias
    )


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        x = self.avg_pool(input)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return input * x


class FPEBlock(nn.Module):
    def __init__(
        self,
        inplanes,
        outplanes,
        dilat,
        downsample=None,
        stride=1,
        t=1,
        scales=4,
        se=False,
        norm_layer=None,
    ):
        super(FPEBlock, self).__init__()
        if inplanes % scales != 0:
            raise ValueError("Planes must be divisible by scales")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        bottleneck_planes = inplanes * t
        self.conv1 = conv1x1(inplanes, bottleneck_planes, stride)
        self.bn1 = norm_layer(bottleneck_planes)
        self.conv2 = nn.ModuleList(
            [
                conv3x3(
                    bottleneck_planes // scales,
                    bottleneck_planes // scales,
                    groups=(bottleneck_planes // scales),
                    dilation=dilat[i],
                    padding=1 * dilat[i],
                )
                for i in range(scales)
            ]
        )
        self.bn2 = nn.ModuleList(
            [norm_layer(bottleneck_planes // scales) for _ in range(scales)]
        )
        self.conv3 = conv1x1(bottleneck_planes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEModule(outplanes) if se else None
        self.downsample = downsample
        self.stride = stride
        self.scales = scales

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        xs = torch.chunk(out, self.scales, 1)
        ys = []
        for s in range(self.scales):
            if s == 0:
                ys.append(self.relu(self.bn2[s](self.conv2[s](xs[s]))))
            else:
                ys.append(
                    self.relu(self.bn2[s](self.conv2[s](xs[s] + ys[-1])))
                )
        out = torch.cat(ys, 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class MEUModule(nn.Module):
    def __init__(self, channels_high, channels_low, channel_out):
        super(MEUModule, self).__init__()

        self.conv1x1_low = nn.Conv2d(
            channels_low, channel_out, kernel_size=1, bias=False
        )
        self.bn_low = nn.BatchNorm2d(channel_out)
        self.sa_conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)

        self.conv1x1_high = nn.Conv2d(
            channels_high, channel_out, kernel_size=1, bias=False
        )
        self.bn_high = nn.BatchNorm2d(channel_out)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca_conv = nn.Conv2d(
            channel_out, channel_out, kernel_size=1, bias=False
        )

        self.sa_sigmoid = nn.Sigmoid()
        self.ca_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low):
        """
        :param fms_high:  High level Feature map. Tensor.
        :param fms_low: Low level Feature map. Tensor.
        """
        _, _, h, w = fms_low.shape

        #
        fms_low = self.conv1x1_low(fms_low)
        fms_low = self.bn_low(fms_low)
        sa_avg_out = self.sa_sigmoid(
            self.sa_conv(torch.mean(fms_low, dim=1, keepdim=True))
        )

        #
        fms_high = self.conv1x1_high(fms_high)
        fms_high = self.bn_high(fms_high)
        ca_avg_out = self.ca_sigmoid(
            self.relu(self.ca_conv(self.avg_pool(fms_high)))
        )

        #
        fms_high_up = F.interpolate(
            fms_high, size=(h, w), mode="bilinear", align_corners=True
        )
        fms_sa_att = sa_avg_out * fms_high_up
        #
        fms_ca_att = ca_avg_out * fms_low

        out = fms_ca_att + fms_sa_att

        return out


class FPENet(nn.Module):
    def __init__(
        self,
        classes=19,
        zero_init_residual=False,
        width=16,
        scales=4,
        se=False,
        norm_layer=None,
    ):
        super(FPENet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        outplanes = [
            int(width * 2**i) for i in range(3)
        ]  # planes=[16,32,64]

        self.block_num = [1, 3, 9]
        self.dilation = [1, 2, 4, 8]

        self.inplanes = outplanes[0]
        self.conv1 = nn.Conv2d(
            1, outplanes[0], kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn1 = norm_layer(outplanes[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(
            FPEBlock,
            outplanes[0],
            self.block_num[0],
            dilation=self.dilation,
            stride=1,
            t=1,
            scales=scales,
            se=se,
            norm_layer=norm_layer,
        )
        self.layer2 = self._make_layer(
            FPEBlock,
            outplanes[1],
            self.block_num[1],
            dilation=self.dilation,
            stride=2,
            t=4,
            scales=scales,
            se=se,
            norm_layer=norm_layer,
        )
        self.layer3 = self._make_layer(
            FPEBlock,
            outplanes[2],
            self.block_num[2],
            dilation=self.dilation,
            stride=2,
            t=4,
            scales=scales,
            se=se,
            norm_layer=norm_layer,
        )
        self.meu1 = MEUModule(64, 32, 64)
        self.meu2 = MEUModule(64, 16, 32)

        # Projection layer
        self.project_layer = nn.Conv2d(32, classes, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, FPEBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def _make_layer(
        self,
        block,
        planes,
        blocks,
        dilation,
        stride=1,
        t=1,
        scales=4,
        se=False,
        norm_layer=None,
    ):
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                dilat=dilation,
                downsample=downsample,
                stride=stride,
                t=t,
                scales=scales,
                se=se,
                norm_layer=norm_layer,
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    dilat=dilation,
                    scales=scales,
                    se=se,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        # stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x_1 = self.layer1(x)

        # stage 2
        x_2_0 = self.layer2[0](x_1)
        x_2_1 = self.layer2[1](x_2_0)
        x_2_2 = self.layer2[2](x_2_1)
        x_2 = x_2_0 + x_2_2

        # stage 3
        x_3_0 = self.layer3[0](x_2)
        x_3_1 = self.layer3[1](x_3_0)
        x_3_2 = self.layer3[2](x_3_1)
        x_3_3 = self.layer3[3](x_3_2)
        x_3_4 = self.layer3[4](x_3_3)
        x_3_5 = self.layer3[5](x_3_4)
        x_3_6 = self.layer3[6](x_3_5)
        x_3_7 = self.layer3[7](x_3_6)
        x_3_8 = self.layer3[8](x_3_7)
        x_3 = x_3_0 + x_3_8

        x2 = self.meu1(x_3, x_2)

        x1 = self.meu2(x2, x_1)

        output = self.project_layer(x1)

        # Bilinear interpolation x2
        output = F.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return F.sigmoid(output)


class CBR(nn.Module):
    """
    This class defines the convolution layer with batch normalization and PReLU activation
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class BR(nn.Module):
    """
    This class groups the batch normalization and PReLU activation
    """

    def __init__(self, nOut):
        """
        :param nOut: output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        :param input: input feature map
        :return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output


class CB(nn.Module):
    """
    This class groups the convolution and batch normalization
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class C(nn.Module):
    """
    This class is for a convolutional layer.
    """

    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class CDilated(nn.Module):
    """
    This class defines the dilated convolution, which can maintain feature map size
    """

    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
            dilation=d,
        )

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        output = self.conv(input)
        return output


class DownSamplerB(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = C(nIn, n, 3, 2)
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-3)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output1 = self.c1(input)
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced
        # by the large effective receptive filed of the ESP module
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        # combine_in_out = input + combine  #shotcut path
        output = self.bn(combine)
        output = self.act(output)
        return output


# ESP block
class DilatedParllelResidualBlockB(nn.Module):
    """
    This class defines the ESP block, which is based on the following principle
        Reduce ---> Split ---> Transform --> Merge
    """

    def __init__(self, nIn, nOut, add=True):
        """
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param add: if true, add a residual connection through identity operation. You can use projection too as
                in ResNet paper, but we avoid to use it if the dimensions are not the same because we do not want to
                increase the module complexity
        """
        super().__init__()
        n = int(nOut / 5)  # K=5,
        n = 1
        n1 = (
            nOut - 4 * n
        )  # (N-(K-1)INT(N/K)) for dilation rate of 2^0, for producing an output feature map of channel=nOut
        # print(n,n1,nOut)
        n1 = 1
        self.c1 = C(
            nIn, n, 1, 1
        )  # the point-wise convolutions with 1x1 help in reducing the computation, channel=c

        # K=5, dilation rate: 2^{k-1},k={1,2,3,...,K}
        self.d1 = CDilated(n, n1, 3, 1, 1)  # dilation rate of 2^0
        self.d2 = CDilated(n, n, 3, 1, 2)  # dilation rate of 2^1
        self.d4 = CDilated(n, n, 3, 1, 4)  # dilation rate of 2^2
        self.d8 = CDilated(n, n, 3, 1, 8)  # dilation rate of 2^3
        self.d16 = CDilated(n, n, 3, 1, 16)  # dilation rate of 2^4
        self.bn = BR(nOut)
        self.add = add

    def forward(self, input):
        """
        :param input: input feature map
        :return: transformed feature map
        """
        # reduce
        output1 = self.c1(input)
        # split and transform
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)

        # Using hierarchical feature fusion (HFF) to ease the gridding artifacts which is introduced
        # by the large effective receptive filed of the ESP module
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16

        # merge
        combine = torch.cat([d1, add1, add2, add3, add4], 1)

        # if residual version
        if self.add:
            combine = input + combine
        output = self.bn(combine)
        return output


class InputProjectionA(nn.Module):
    """
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3, for input reinforcement, which establishes a direct link between
    the input image and encoding stage, improving the flow of information.
    """

    def __init__(self, samplingTimes):
        """
        :param samplingTimes: The rate at which you want to down-sample the image
        """
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        """
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        """
        for pool in self.pool:
            input = pool(input)
        return input


class ESPNet_Encoder(nn.Module):
    """
    This class defines the ESPNet-C network in the paper
    """

    def __init__(self, classes=19, p=5, q=3):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        """
        super().__init__()
        self.level1 = CBR(
            1, 16, 3, 2
        )  # feature map size divided 2,                         1/2
        self.sample1 = InputProjectionA(
            1
        )  # down-sample for input reinforcement, factor=2
        self.sample2 = InputProjectionA(
            2
        )  # down-sample for input reinforcement, factor=4

        self.b1 = BR(16 + 1)
        self.level2_0 = DownSamplerB(
            16 + 1, 64
        )  # Downsample Block, feature map size divided 2,    1/4

        self.level2 = nn.ModuleList()
        for i in range(0, p):
            self.level2.append(
                DilatedParllelResidualBlockB(64, 64)
            )  # ESP block
        self.b2 = BR(128 + 1)

        self.level3_0 = DownSamplerB(
            128 + 1, 128
        )  # Downsample Block, feature map size divided 2,   1/8
        self.level3 = nn.ModuleList()
        for i in range(0, q):
            self.level3.append(
                DilatedParllelResidualBlockB(128, 128)
            )  # ESPblock
        self.b3 = BR(256)

        self.classifier = C(256, classes, 1, 1)

    def forward(self, input):
        """
        :param input: Receives the input RGB image
        :return: the transformed feature map with spatial dimensions 1/8th of the input image
        """
        output0 = self.level1(input)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.b2(torch.cat([output1, output1_0, inp2], 1))

        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.b3(torch.cat([output2_0, output2], 1))

        classifier = self.classifier(output2_cat)

        # return classifier
        out = F.upsample(
            classifier, input.size()[2:], mode="bilinear"
        )  # Upsample score map, factor=8
        return out


class ESPNet(nn.Module):
    """
    This class defines the ESPNet network
    """

    def __init__(self, classes=19, p=2, q=3, encoderFile=None):
        """
        :param classes: number of classes in the dataset. Default is 20 for the cityscapes
        :param p: depth multiplier
        :param q: depth multiplier
        :param encoderFile: pretrained encoder weights. Recall that we first trained the ESPNet-C and then attached the
                            RUM-based light weight decoder. See paper for more details.
        """
        super().__init__()
        self.encoder = ESPNet_Encoder(classes, p, q)
        if encoderFile is not None:
            self.encoder.load_state_dict(torch.load(encoderFile))
            print("Encoder loaded!")
        # load the encoder modules
        self.en_modules = []
        for i, m in enumerate(self.encoder.children()):
            self.en_modules.append(m)

        # light-weight decoder
        self.level3_C = C(128 + 1, classes, 1, 1)
        self.br = nn.BatchNorm2d(classes, eps=1e-03)
        self.conv = CBR(19 + classes, classes, 3, 1)

        self.up_l3 = nn.Sequential(
            nn.ConvTranspose2d(
                classes,
                classes,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            )
        )
        self.combine_l2_l3 = nn.Sequential(
            BR(2 * classes),
            DilatedParllelResidualBlockB(2 * classes, classes, add=False),
        )

        self.up_l2 = nn.Sequential(
            nn.ConvTranspose2d(
                classes,
                classes,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=False,
            ),
            BR(classes),
        )

        self.classifier = nn.ConvTranspose2d(
            classes,
            classes,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=False,
        )

    def forward(self, input):
        """
        :param input: RGB image
        :return: transformed feature map
        """
        output0 = self.en_modules[0](input)
        inp1 = self.en_modules[1](input)
        inp2 = self.en_modules[2](input)

        output0_cat = self.en_modules[3](torch.cat([output0, inp1], 1))
        output1_0 = self.en_modules[4](output0_cat)  # down-sampled

        for i, layer in enumerate(self.en_modules[5]):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.en_modules[6](
            torch.cat([output1, output1_0, inp2], 1)
        )

        output2_0 = self.en_modules[7](output1_cat)  # down-sampled
        for i, layer in enumerate(self.en_modules[8]):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.en_modules[9](
            torch.cat([output2_0, output2], 1)
        )  # concatenate for feature map width expansion

        output2_c = self.up_l3(
            self.br(self.en_modules[10](output2_cat))
        )  # RUM

        output1_C = self.level3_C(
            output1_cat
        )  # project to C-dimensional space
        print(output1_C.shape, output2_c.shape)
        comb_l2_l3 = self.up_l2(
            self.combine_l2_l3(torch.cat([output1_C, output2_c], 1))
        )  # RUM

        concat_features = self.conv(torch.cat([comb_l2_l3, output0_cat], 1))

        classifier = self.classifier(concat_features)

        return F.sigmoid(classifier)


class DownsamplerBlockESNET(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            ninput,
            noutput,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)

        return F.relu(output)


class FCU(nn.Module):
    def __init__(self, chann, kernel_size, dropprob, dilated):
        """
        Factorized Convolution Unit

        """
        super(FCU, self).__init__()

        padding = int((kernel_size - 1) // 2) * dilated

        self.conv3x1_1 = nn.Conv2d(
            chann,
            chann,
            (kernel_size, 1),
            stride=1,
            padding=(int((kernel_size - 1) // 2) * 1, 0),
            bias=True,
        )

        self.conv1x3_1 = nn.Conv2d(
            chann,
            chann,
            (1, kernel_size),
            stride=1,
            padding=(0, int((kernel_size - 1) // 2) * 1),
            bias=True,
        )

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (kernel_size, 1),
            stride=1,
            padding=(padding, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, kernel_size),
            stride=1,
            padding=(0, padding),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        residual = input
        output = self.conv3x1_1(input)
        output = self.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = self.relu(output)

        output = self.conv3x1_2(output)
        output = self.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(residual + output, inplace=True)


class PFCU(nn.Module):
    def __init__(self, chann):
        """
        Parallel Factorized Convolution Unit

        """

        super(PFCU, self).__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_22 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(2, 0),
            bias=True,
            dilation=(2, 1),
        )
        self.conv1x3_22 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 2),
            bias=True,
            dilation=(1, 2),
        )

        self.conv3x1_25 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(5, 0),
            bias=True,
            dilation=(5, 1),
        )
        self.conv1x3_25 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 5),
            bias=True,
            dilation=(1, 5),
        )

        self.conv3x1_29 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(9, 0),
            bias=True,
            dilation=(9, 1),
        )
        self.conv1x3_29 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 9),
            bias=True,
            dilation=(1, 9),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(0.3)

    def forward(self, input):
        residual = input
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output2 = self.conv3x1_22(output)
        output2 = F.relu(output2)
        output2 = self.conv1x3_22(output2)
        output2 = self.bn2(output2)
        if self.dropout.p != 0:
            output2 = self.dropout(output2)

        output5 = self.conv3x1_25(output)
        output5 = F.relu(output5)
        output5 = self.conv1x3_25(output5)
        output5 = self.bn2(output5)
        if self.dropout.p != 0:
            output5 = self.dropout(output5)

        output9 = self.conv3x1_29(output)
        output9 = F.relu(output9)
        output9 = self.conv1x3_29(output9)
        output9 = self.bn2(output9)
        if self.dropout.p != 0:
            output9 = self.dropout(output9)

        return F.relu(residual + output2 + output5 + output9, inplace=True)


class ESNet(nn.Module):
    def __init__(self, classes):
        super().__init__()
        # -----ESNET---------#
        self.initial_block = DownsamplerBlockESNET(1, 16)

        self.layers = nn.ModuleList()

        for x in range(0, 3):
            self.layers.append(FCU(16, 3, 0.03, 1))

        self.layers.append(DownsamplerBlockESNET(16, 64))

        for x in range(0, 2):
            self.layers.append(FCU(64, 5, 0.03, 1))

        self.layers.append(DownsamplerBlockESNET(64, 128))

        for x in range(0, 3):
            self.layers.append(PFCU(chann=128))

        self.layers.append(UpsamplerBlock(128, 64))
        self.layers.append(FCU(64, 5, 0, 1))
        self.layers.append(FCU(64, 5, 0, 1))

        self.layers.append(UpsamplerBlock(64, 16))
        self.layers.append(FCU(16, 3, 0, 1))
        self.layers.append(FCU(16, 3, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16, classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return F.sigmoid(output)


class DownsamplerBlockERFNet(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv2d(
            chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True
        )

        self.conv1x3_1 = nn.Conv2d(
            chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True
        )

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1),
        )

        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated),
        )

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        return F.relu(
            output + input
        )  # +input = identity (residual connection)


class EncoderERFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.initial_block = DownsamplerBlockERFNet(1, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlockERFNet(16, 64))

        for x in range(0, 5):  # 5 times
            self.layers.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers.append(DownsamplerBlockERFNet(64, 128))

        for x in range(0, 2):  # 2 times
            self.layers.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers.append(non_bottleneck_1d(128, 0.3, 16))

        # Only in encoder mode:
        self.output_conv = nn.Conv2d(
            128, num_classes, 1, stride=1, padding=0, bias=True
        )

    def forward(self, input, predict=False):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        if predict:
            output = self.output_conv(output)

        return output


class UpsamplerBlockERFNet(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            ninput,
            noutput,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=True,
        )
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class DecoderERFNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.layers = nn.ModuleList()

        self.layers.append(UpsamplerBlockERFNet(128, 64))
        self.layers.append(non_bottleneck_1d(64, 0, 1))
        self.layers.append(non_bottleneck_1d(64, 0, 1))

        self.layers.append(UpsamplerBlockERFNet(64, 16))
        self.layers.append(non_bottleneck_1d(16, 0, 1))
        self.layers.append(non_bottleneck_1d(16, 0, 1))

        self.output_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            2,
            stride=2,
            padding=0,
            output_padding=0,
            bias=True,
        )

    def forward(self, input):
        output = input

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output


# ERFNet
class ERFNet(nn.Module):
    def __init__(
        self, classes, encoder=None
    ):  # use encoder to pass pretrained encoder
        super().__init__()

        if encoder is None:
            self.encoder = EncoderERFNet(classes)
        else:
            self.encoder = encoder
        self.decoder = DecoderERFNet(classes)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)  # predict=False by default
            return F.sigmoid(self.decoder.forward(output))


class InitialBlockENet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        bias=False,
        relu=True,
    ):
        super(InitialBlockENet, self).__init__()

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 1,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            bias=bias,
        )
        # MP need padding too
        self.ext_branch = nn.MaxPool2d(kernel_size, stride=2, padding=padding)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_prelu = activation

    def forward(self, input):
        main = self.main_branch(input)
        ext = self.ext_branch(input)

        out = torch.cat((main, ext), dim=1)

        out = self.batch_norm(out)
        return self.out_prelu(out)


class RegularBottleneckENet(nn.Module):
    def __init__(
        self,
        channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dilation=1,
        asymmetric=False,
        dropout_prob=0.0,
        bias=False,
        relu=True,
    ):
        super(RegularBottleneckENet, self).__init__()

        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # 1x1 projection conv
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                channels, internal_channels, kernel_size=1, stride=1, bias=bias
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation,
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation,
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                nn.BatchNorm2d(internal_channels),
                activation,
            )

        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, channels, kernel_size=1, stride=1, bias=bias
            ),
            nn.BatchNorm2d(channels),
            activation,
        )
        self.ext_regu1 = nn.Dropout2d(p=dropout_prob)
        self.out_prelu = activation

    def forward(self, input):
        main = input

        ext = self.ext_conv1(input)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regu1(ext)

        out = main + ext
        return self.out_prelu(out)


class DownsamplingBottleneckENet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        return_indices=False,
        dropout_prob=0.0,
        bias=False,
        relu=True,
    ):
        super().__init__()

        # Store parameters that are needed later
        self.return_indices = return_indices

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_max1 = nn.MaxPool2d(
            kernel_size,
            stride=2,
            padding=padding,
            return_indices=return_indices,
        )

        # Extension branch - 2x2 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 2x2 projection convolution with stride 2, no padding
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # Convolution
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x):
        # Main branch shortcut
        if self.return_indices:
            main, max_indices = self.main_max1(x)
        else:
            main = self.main_max1(x)

        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Main branch channel padding
        # calculate for padding ch_ext - ch_main
        n, ch_ext, h, w = ext.size()
        ch_main = main.size()[1]
        padding = torch.zeros(n, ch_ext - ch_main, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        # convert padding accordingly
        if main.is_cuda:
            padding = padding.cuda()

        # Concatenate, padding for less channels of main branch
        main = torch.cat((main, padding), 1)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out), max_indices


class UpsamplingBottleneckENet(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        internal_ratio=4,
        kernel_size=3,
        padding=0,
        dropout_prob=0.0,
        bias=False,
        relu=True,
    ):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU()
        else:
            activation = nn.PReLU()

        # Main branch - max pooling followed by feature map (channels) padding
        self.main_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

        # Remember that the stride is the same as the kernel_size, just like
        # the max pooling layers
        self.main_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Extension branch - 1x1 convolution, followed by a regular, dilated or
        # asymmetric convolution, followed by another 1x1 convolution. Number
        # of channels is doubled.

        # 1x1 projection convolution with stride 1
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels, internal_channels, kernel_size=1, bias=bias
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # Transposed convolution
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                internal_channels,
                internal_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=1,
                bias=bias,
            ),
            nn.BatchNorm2d(internal_channels),
            activation,
        )

        # 1x1 expansion convolution
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, out_channels, kernel_size=1, bias=bias
            ),
            nn.BatchNorm2d(out_channels),
            activation,
        )

        self.ext_regul = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_prelu = activation

    def forward(self, x, max_indices):
        # Main branch shortcut
        main = self.main_conv1(x)
        main = self.main_unpool1(main, max_indices)
        # Extension branch
        ext = self.ext_conv1(x)
        ext = self.ext_conv2(ext)
        ext = self.ext_conv3(ext)
        ext = self.ext_regul(ext)

        # Add main and extension branches
        out = main + ext

        return self.out_prelu(out)


class ENet(nn.Module):
    def __init__(self, classes, encoder_relu=False, decoder_relu=True):
        super().__init__()
        # source code
        self.name = "BaseLine_ENet_trans"

        self.initial_block = InitialBlockENet(
            1, 16, kernel_size=3, padding=1, relu=encoder_relu
        )

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneckENet(
            16,
            64,
            padding=1,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu,
        )
        self.regular1_1 = RegularBottleneckENet(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu
        )
        self.regular1_2 = RegularBottleneckENet(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu
        )
        self.regular1_3 = RegularBottleneckENet(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu
        )
        self.regular1_4 = RegularBottleneckENet(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu
        )

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneckENet(
            64,
            128,
            padding=1,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu,
        )
        self.regular2_1 = RegularBottleneckENet(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu
        )
        self.dilated2_2 = RegularBottleneckENet(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu
        )
        self.asymmetric2_3 = RegularBottleneckENet(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu,
        )
        self.dilated2_4 = RegularBottleneckENet(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu
        )
        self.regular2_5 = RegularBottleneckENet(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu
        )
        self.dilated2_6 = RegularBottleneckENet(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu
        )
        self.asymmetric2_7 = RegularBottleneckENet(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,
        )
        self.dilated2_8 = RegularBottleneckENet(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu
        )

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneckENet(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu
        )
        self.dilated3_1 = RegularBottleneckENet(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu
        )
        self.asymmetric3_2 = RegularBottleneckENet(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu,
        )
        self.dilated3_3 = RegularBottleneckENet(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu
        )
        self.regular3_4 = RegularBottleneckENet(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu
        )
        self.dilated3_5 = RegularBottleneckENet(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu
        )
        self.asymmetric3_6 = RegularBottleneckENet(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu,
        )
        self.dilated3_7 = RegularBottleneckENet(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu
        )

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneckENet(
            128, 64, padding=1, dropout_prob=0.1, relu=decoder_relu
        )
        self.regular4_1 = RegularBottleneckENet(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu
        )
        self.regular4_2 = RegularBottleneckENet(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu
        )

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneckENet(
            64, 16, padding=1, dropout_prob=0.1, relu=decoder_relu
        )
        self.regular5_1 = RegularBottleneckENet(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu
        )
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            classes,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )

        self.project_layer = nn.Conv2d(128, classes, 1, bias=False)

    def forward(self, x):
        # Initial block
        x = self.initial_block(x)

        # Stage 1 - Encoder
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # x = self.project_layer(x)
        # x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=True)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)

        return F.sigmoid(x)


class DownsamplerBlockEDANet(nn.Module):
    def __init__(self, ninput, noutput):
        super(DownsamplerBlockEDANet, self).__init__()

        self.ninput = ninput
        self.noutput = noutput

        if self.ninput < self.noutput:
            # Wout > Win
            self.conv = nn.Conv2d(
                ninput, noutput - ninput, kernel_size=3, stride=2, padding=1
            )
            self.pool = nn.MaxPool2d(2, stride=2)
        else:
            # Wout < Win
            self.conv = nn.Conv2d(
                ninput, noutput, kernel_size=3, stride=2, padding=1
            )

        self.bn = nn.BatchNorm2d(noutput)

    def forward(self, x):
        if self.ninput < self.noutput:
            output = torch.cat([self.conv(x), self.pool(x)], 1)
        else:
            output = self.conv(x)

        output = self.bn(output)
        return F.relu(output)


# --- Build the EDANet Module --- #
class EDAModule(nn.Module):
    def __init__(self, ninput, dilated, k=40, dropprob=0.02):
        super().__init__()

        # k: growthrate
        # dropprob:a dropout layer between the last ReLU and the concatenation of each module

        self.conv1x1 = nn.Conv2d(ninput, k, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(k)

        self.conv3x1_1 = nn.Conv2d(k, k, kernel_size=(3, 1), padding=(1, 0))
        self.conv1x3_1 = nn.Conv2d(k, k, kernel_size=(1, 3), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(k)

        self.conv3x1_2 = nn.Conv2d(
            k, k, (3, 1), stride=1, padding=(dilated, 0), dilation=dilated
        )
        self.conv1x3_2 = nn.Conv2d(
            k, k, (1, 3), stride=1, padding=(0, dilated), dilation=dilated
        )
        self.bn2 = nn.BatchNorm2d(k)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, x):
        input = x

        output = self.conv1x1(x)
        output = self.bn0(output)
        output = F.relu(output)

        output = self.conv3x1_1(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)
        output = F.relu(output)

        if self.dropout.p != 0:
            output = self.dropout(output)

        output = torch.cat([output, input], 1)
        # print output.size() #check the output
        return output


# --- Build the EDANet Block --- #
class EDANetBlock(nn.Module):
    def __init__(self, in_channels, num_dense_layer, dilated, growth_rate):
        """
        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super().__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(EDAModule(_in_channels, dilated[i], growth_rate))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)
        # self.conv_1x1 = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)

    def forward(self, x):
        out = self.residual_dense_layers(x)
        # out = self.conv_1x1(out)
        # out = out + x
        return out


class EDANet(nn.Module):
    def __init__(self, classes=19):
        super(EDANet, self).__init__()

        self.layers = nn.ModuleList()

        # DownsamplerBlock1
        self.layers.append(DownsamplerBlockEDANet(1, 15))

        # DownsamplerBlock2
        self.layers.append(DownsamplerBlockEDANet(15, 60))

        # EDA Block1
        self.layers.append(EDANetBlock(60, 5, [1, 1, 1, 2, 2], 40))

        # DownsamplerBlock3
        self.layers.append(DownsamplerBlockEDANet(260, 130))

        # # EDA Block2
        self.layers.append(EDANetBlock(130, 8, [2, 2, 4, 4, 8, 8, 16, 16], 40))

        # Projection layer
        self.project_layer = nn.Conv2d(450, classes, kernel_size=1)

        self.weights_init()

    def weights_init(self):
        for idx, m in enumerate(self.modules()):
            classname = m.__class__.__name__
            if classname.find("Conv") != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find("BatchNorm") != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        output = x

        for layer in self.layers:
            output = layer(output)

        output = self.project_layer(output)

        # Bilinear interpolation x8
        output = F.interpolate(
            output, scale_factor=8, mode="bilinear", align_corners=True
        )

        return F.sigmoid(output)


class Conv(nn.Module):
    def __init__(
        self,
        nIn,
        nOut,
        kSize,
        stride,
        padding,
        dilation=(1, 1),
        groups=1,
        bn_acti=False,
        bias=False,
    ):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(
            nIn,
            nOut,
            kernel_size=kSize,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)

        self.dconv3x1 = Conv(
            nIn // 2,
            nIn // 2,
            (dkSize, 1),
            1,
            padding=(1, 0),
            groups=nIn // 2,
            bn_acti=True,
        )
        self.dconv1x3 = Conv(
            nIn // 2,
            nIn // 2,
            (1, dkSize),
            1,
            padding=(0, 1),
            groups=nIn // 2,
            bn_acti=True,
        )
        self.ddconv3x1 = Conv(
            nIn // 2,
            nIn // 2,
            (dkSize, 1),
            1,
            padding=(1 * d, 0),
            dilation=(d, 1),
            groups=nIn // 2,
            bn_acti=True,
        )
        self.ddconv1x3 = Conv(
            nIn // 2,
            nIn // 2,
            (1, dkSize),
            1,
            padding=(0, 1 * d),
            dilation=(1, d),
            groups=nIn // 2,
            bn_acti=True,
        )

        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)

        br1 = self.dconv3x1(output)
        br1 = self.dconv1x3(br1)
        br2 = self.ddconv3x1(output)
        br2 = self.ddconv1x3(br2)

        output = br1 + br2
        output = self.bn_relu_2(output)
        output = self.conv1x1(output)

        return output + input


class DownSamplingBlockDABNet(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input


class DABNet(nn.Module):
    def __init__(self, classes=19, block_1=3, block_2=6):
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(1, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times

        self.bn_prelu_1 = BNPReLU(32 + 1)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlockDABNet(32 + 1, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module(
                "DAB_Module_1_" + str(i), DABModule(64, d=2)
            )
        self.bn_prelu_2 = BNPReLU(128 + 1)

        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]
        self.downsample_2 = DownSamplingBlockDABNet(128 + 1, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module(
                "DAB_Module_2_" + str(i), DABModule(128, d=dilation_block_2[i])
            )
        self.bn_prelu_3 = BNPReLU(256 + 1)

        self.classifier = nn.Sequential(Conv(257, classes, 1, 1, padding=0))

    def forward(self, input):
        output0 = self.init_conv(input)

        down_1 = self.down_1(input)
        down_2 = self.down_2(input)
        down_3 = self.down_3(input)

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))

        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(
            torch.cat([output1, output1_0, down_2], 1)
        )

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(
            torch.cat([output2, output2_0, down_3], 1)
        )

        out = self.classifier(output2_cat)
        out = F.interpolate(
            out, input.size()[2:], mode="bilinear", align_corners=False
        )

        return F.sigmoid(out)


class Custom_Conv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        **kwargs
    ):
        super(Custom_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class DepthSepConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthSepConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels,
                dw_channels,
                3,
                stride,
                1,
                groups=dw_channels,
                bias=False,
            ),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class DepthConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(DepthConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                dw_channels,
                out_channels,
                3,
                stride,
                1,
                groups=dw_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneckContextNet(nn.Module):
    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneckContextNet, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            Custom_Conv(in_channels, in_channels * t, 1),
            DepthConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class Shallow_net(nn.Module):
    def __init__(
        self, dw_channels1=32, dw_channels2=64, out_channels=128, **kwargs
    ):
        super(Shallow_net, self).__init__()
        self.conv = Custom_Conv(1, dw_channels1, 3, 2)
        self.dsconv1 = DepthSepConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = DepthSepConv(dw_channels2, out_channels, 2)
        self.dsconv3 = DepthSepConv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dsconv3(x)
        return x


class Deep_net(nn.Module):
    def __init__(self, in_channels, block_channels, t, num_blocks, **kwargs):
        super(Deep_net, self).__init__()
        self.block_channels = block_channels
        self.t = t
        self.num_blocks = num_blocks

        self.conv_ = Custom_Conv(1, in_channels, 3, 2)
        self.bottleneck1 = self._layer(
            LinearBottleneckContextNet,
            in_channels,
            block_channels[0],
            num_blocks[0],
            t[0],
            1,
        )
        self.bottleneck2 = self._layer(
            LinearBottleneckContextNet,
            block_channels[0],
            block_channels[1],
            num_blocks[1],
            t[1],
            1,
        )
        self.bottleneck3 = self._layer(
            LinearBottleneckContextNet,
            block_channels[1],
            block_channels[2],
            num_blocks[2],
            t[2],
            2,
        )
        self.bottleneck4 = self._layer(
            LinearBottleneckContextNet,
            block_channels[2],
            block_channels[3],
            num_blocks[3],
            t[3],
            2,
        )
        self.bottleneck5 = self._layer(
            LinearBottleneckContextNet,
            block_channels[3],
            block_channels[4],
            num_blocks[4],
            t[4],
            1,
        )
        self.bottleneck6 = self._layer(
            LinearBottleneckContextNet,
            block_channels[4],
            block_channels[5],
            num_blocks[5],
            t[5],
            1,
        )

    def _layer(self, block, in_channels, out_channels, blocks, t, stride):
        layers = []
        layers.append(block(in_channels, out_channels, t, stride))
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels, t, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_(x)
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        x = self.bottleneck4(x)
        x = self.bottleneck5(x)
        x = self.bottleneck6(x)
        return x


class FeatureFusionModuleContextNet(nn.Module):
    def __init__(
        self,
        highter_in_channels,
        lower_in_channels,
        out_channels,
        scale_factor=4,
        **kwargs
    ):
        super(FeatureFusionModuleContextNet, self).__init__()
        self.scale_factor = scale_factor
        self.dwconv = DepthConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(highter_in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(True)

    def forward(self, higher_res_feature, lower_res_feature):
        _, _, h, w = higher_res_feature.size()
        lower_res_feature = F.interpolate(
            lower_res_feature, size=(h, w), mode="bilinear", align_corners=True
        )
        lower_res_feature = self.dwconv(lower_res_feature)
        lower_res_feature = self.conv_lower_res(lower_res_feature)

        higher_res_feature = self.conv_higher_res(higher_res_feature)
        out = higher_res_feature + lower_res_feature
        return self.relu(out)


class ClassiferContextNet(nn.Module):
    def __init__(self, dw_channels, num_classes, stride=1, **kwargs):
        super(ClassiferContextNet, self).__init__()
        self.dsconv1 = DepthSepConv(dw_channels, dw_channels, stride)
        self.dsconv2 = DepthSepConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1), nn.Conv2d(dw_channels, num_classes, 1)
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.conv(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, classes, aux=False, **kwargs):
        super(ContextNet, self).__init__()
        self.aux = aux
        self.spatial_detail = Shallow_net(32, 64, 128)
        self.context_feature_extractor = Deep_net(
            32,
            [32, 32, 48, 64, 96, 128],
            [1, 6, 6, 6, 6, 6],
            [1, 1, 3, 3, 2, 2],
        )
        self.feature_fusion = FeatureFusionModuleContextNet(128, 128, 128)
        self.classifier = ClassiferContextNet(128, classes)
        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(128, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(True),
                nn.Dropout(0.1),
                nn.Conv2d(32, classes, 1),
            )

    def forward(self, x):
        size = x.size()[2:]

        higher_res_features = self.spatial_detail(x)

        x_low = F.interpolate(
            x, scale_factor=0.25, mode="bilinear", align_corners=True
        )

        x = self.context_feature_extractor(x_low)

        x = self.feature_fusion(higher_res_features, x)

        x = self.classifier(x)

        outputs = []
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)

        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(higher_res_features)
            auxout = F.interpolate(
                auxout, size, mode="bilinear", align_corners=True
            )
            outputs.append(auxout)

        return F.sigmoid(x)


class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optinal stide for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        return output


class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            groups=nIn,
            bias=False,
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class DilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
            dilation=d,
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            groups=nIn,
            bias=False,
            dilation=d,
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """

    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ConvCGNet(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(
            nIn,
            nOut,
            (kSize, kSize),
            stride=stride,
            padding=(padding, padding),
            bias=False,
        )

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output


class ContextGuidedBlock_Down(nn.Module):
    """
    the size of feature map divided 2, (H,W,C)---->(H/2, W/2, 2C)
    """

    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16):
        """
        args:
           nIn: the channel of input feature map
           nOut: the channel of output feature map, and nOut=2*nIn
        """
        super().__init__()
        self.conv1x1 = ConvBNPReLU(
            nIn, nOut, 3, 2
        )  # size/2, channel: nIn--->nOut

        self.F_loc = ChannelWiseConv(nOut, nOut, 3, 1)
        self.F_sur = ChannelWiseDilatedConv(nOut, nOut, 3, 1, dilation_rate)

        self.bn = nn.BatchNorm2d(2 * nOut, eps=1e-3)
        self.act = nn.PReLU(2 * nOut)
        self.reduce = ConvCGNet(
            2 * nOut, nOut, 1, 1
        )  # reduce dimension: 2*nOut--->nOut

        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)  # the joint feature
        joi_feat = self.bn(joi_feat)
        joi_feat = self.act(joi_feat)
        joi_feat = self.reduce(joi_feat)  # channel= nOut

        output = self.F_glo(
            joi_feat
        )  # F_glo is employed to refine the joint feature

        return output


class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=2, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels,
           add: if true, residual learning
        """
        super().__init__()
        n = int(nOut / 2)
        self.conv1x1 = ConvBNPReLU(
            nIn, n, 1, 1
        )  # 1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1)  # local feature
        self.F_sur = ChannelWiseDilatedConv(
            n, n, 3, 1, dilation_rate
        )  # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo = FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur = self.F_sur(output)

        joi_feat = torch.cat([loc, sur], 1)

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(
            joi_feat
        )  # F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output = input + output
        return output


class CGNet(nn.Module):
    """
    This class defines the proposed Context Guided Network (CGNet) in this work.
    """

    def __init__(self, classes=19, M=3, N=21, dropout_flag=False):
        """
        args:
          classes: number of classes in the dataset. Default is 19 for the cityscapes
          M: the number of blocks in stage 2
          N: the number of blocks in stage 3
        """
        super().__init__()
        self.level1_0 = ConvBNPReLU(
            1, 32, 3, 2
        )  # feature map size divided 2, 1/2
        self.level1_1 = ConvBNPReLU(32, 32, 3, 1)
        self.level1_2 = ConvBNPReLU(32, 32, 3, 1)

        self.sample1 = InputInjection(
            1
        )  # down-sample for Input Injection, factor=2
        self.sample2 = InputInjection(
            2
        )  # down-sample for Input Injiection, factor=4

        self.b1 = BNPReLU(32 + 1)

        # stage 2
        self.level2_0 = ContextGuidedBlock_Down(
            32 + 1, 64, dilation_rate=2, reduction=8
        )
        self.level2 = nn.ModuleList()
        for i in range(0, M - 1):
            self.level2.append(
                ContextGuidedBlock(64, 64, dilation_rate=2, reduction=8)
            )  # CG block
        self.bn_prelu_2 = BNPReLU(128 + 1)

        # stage 3
        self.level3_0 = ContextGuidedBlock_Down(
            128 + 1, 128, dilation_rate=4, reduction=16
        )
        self.level3 = nn.ModuleList()
        for i in range(0, N - 1):
            self.level3.append(
                ContextGuidedBlock(128, 128, dilation_rate=4, reduction=16)
            )  # CG block
        self.bn_prelu_3 = BNPReLU(256)

        if dropout_flag:
            print("have droput layer")
            self.classifier = nn.Sequential(
                nn.Dropout2d(0.1, False), ConvCGNet(256, classes, 1, 1)
            )
        else:
            self.classifier = nn.Sequential(ConvCGNet(256, classes, 1, 1))

        # init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find("Conv2d") != -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                elif classname.find("ConvTranspose2d") != -1:
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, input):
        """
        args:
            input: Receives the input RGB image
            return: segmentation map
        """
        # stage 1
        output0 = self.level1_0(input)
        output0 = self.level1_1(output0)
        output0 = self.level1_2(output0)
        inp1 = self.sample1(input)
        inp2 = self.sample2(input)

        # stage 2
        output0_cat = self.b1(torch.cat([output0, inp1], 1))
        output1_0 = self.level2_0(output0_cat)  # down-sampled

        for i, layer in enumerate(self.level2):
            if i == 0:
                output1 = layer(output1_0)
            else:
                output1 = layer(output1)

        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, inp2], 1))

        # stage 3
        output2_0 = self.level3_0(output1_cat)  # down-sampled
        for i, layer in enumerate(self.level3):
            if i == 0:
                output2 = layer(output2_0)
            else:
                output2 = layer(output2)

        output2_cat = self.bn_prelu_3(torch.cat([output2_0, output2], 1))

        # classifier
        classifier = self.classifier(output2_cat)

        # upsample segmenation map ---> the input image size
        out = F.interpolate(
            classifier, input.size()[2:], mode="bilinear", align_corners=False
        )  # Upsample score map, factor=8
        return F.sigmoid(out)
