from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import functional as F
import math


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


class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
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
            SELayer = SEModule
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
        inputs_shape=(3, 224, 224),
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
        return F.softmax(out_conv, dim=1)

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
