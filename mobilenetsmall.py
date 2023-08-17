#based on: https://github.com/Isnot2bad/Micro-Net

"""Searching for MobileNetV3"""
import torch.nn as nn

__all__ = ['MobileNetV3', 'get_mobilenet_v3', 'mobilenet_v3_small_1_0']

import torch
import torch.nn as nn

__all__ = ['Hswish', 'ConvBNHswish', 'Bottleneck', 'SEModule']


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return x * self.relu6(x + 3.) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.relu6 = nn.ReLU6(inplace)

    def forward(self, x):
        return self.relu6(x + 3.) / 6.


class ConvBNHswish(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ConvBNHswish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
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
            Hsigmoid(True)
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
    def __init__(self, in_channels, out_channels, exp_size, kernel_size, stride, dilation=1, se=False, nl='RE',
                 norm_layer=nn.BatchNorm2d, **kwargs):
        super(Bottleneck, self).__init__()
        assert stride in [1, 2]
        self.use_res_connect = stride == 1 and in_channels == out_channels
        if nl == 'HS':
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
            nn.Conv2d(exp_size, exp_size, kernel_size, stride, (kernel_size - 1) // 2 * dilation,
                      dilation, groups=exp_size, bias=False),
            norm_layer(exp_size),
            SELayer(exp_size),
            act(True),
            # pw-linear
            nn.Conv2d(exp_size, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


if __name__ == '__main__':
    img = torch.randn(1, 16, 64, 64)
    model = Bottleneck(16, 16, 16, 3, 1)
    out = model(img)
    print(out.size())







class MobileNetV3(nn.Module):
    def __init__(self, nclass=2, mode='small', width_mult=1.0, dilated=False, norm_layer=nn.BatchNorm2d):
        super(MobileNetV3, self).__init__()
        if mode == 'large':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, False, 'RE', 1],
                [3, 64, 24, False, 'RE', 2],
                [3, 72, 24, False, 'RE', 1], ]
            layer2_setting = [
                [5, 72, 40, True, 'RE', 2],
                [5, 120, 40, True, 'RE', 1],
                [5, 120, 40, True, 'RE', 1], ]
            layer3_setting = [
                [3, 240, 80, False, 'HS', 2],
                [3, 200, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 184, 80, False, 'HS', 1],
                [3, 480, 112, True, 'HS', 1],
                [3, 672, 112, True, 'HS', 1],
                [5, 672, 112, True, 'HS', 1], ]
            layer4_setting = [
                [5, 672, 160, True, 'HS', 2],
                [5, 960, 160, True, 'HS', 1], ]
        elif mode == 'small':
            layer1_setting = [
                # k, exp_size, c, se, nl, s
                [3, 16, 16, True, 'RE', 2], ]
            layer2_setting = [
                [3, 72, 24, False, 'RE', 2],
                [3, 88, 24, False, 'RE', 1], ]
            layer3_setting = [
                [5, 96, 40, True, 'HS', 2],
                [5, 240, 40, True, 'HS', 1],
                [5, 240, 40, True, 'HS', 1],
                [5, 120, 48, True, 'HS', 1],
                [5, 144, 48, True, 'HS', 1], ]
            layer4_setting = [
                [5, 288, 96, True, 'HS', 2],
                [5, 576, 96, True, 'HS', 1],
                [5, 576, 96, True, 'HS', 1], ]
        else:
            raise ValueError('Unknown mode.')

        # building first layer
        input_channels = int(16 * width_mult) if width_mult > 1.0 else 16
        self.conv1 = ConvBNHswish(3, input_channels, 3, 2, 1, norm_layer=norm_layer)

        # building bottleneck blocks
        self.layer1, input_channels = self._make_layer(Bottleneck, input_channels, layer1_setting,
                                                       width_mult, norm_layer=norm_layer)
        self.layer2, input_channels = self._make_layer(Bottleneck, input_channels, layer2_setting,
                                                       width_mult, norm_layer=norm_layer)
        self.layer3, input_channels = self._make_layer(Bottleneck, input_channels, layer3_setting,
                                                       width_mult, norm_layer=norm_layer)
        if dilated:
            self.layer4, input_channels = self._make_layer(Bottleneck, input_channels, layer4_setting,
                                                           width_mult, dilation=2, norm_layer=norm_layer)
        else:
            self.layer4, input_channels = self._make_layer(Bottleneck, input_channels, layer4_setting,
                                                           width_mult, norm_layer=norm_layer)

        # building last several layers
        classifier = list()
        if mode == 'large':
            last_bneck_channels = int(960 * width_mult) if width_mult > 1.0 else 960
            self.conv5 = ConvBNHswish(input_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        elif mode == 'small':
            last_bneck_channels = int(576 * width_mult) if width_mult > 1.0 else 576
            self.conv5 = ConvBNHswish(input_channels, last_bneck_channels, 1, norm_layer=norm_layer)
            classifier.append(SEModule(last_bneck_channels))
            classifier.append(nn.AdaptiveAvgPool2d(1))
            classifier.append(nn.Conv2d(last_bneck_channels, 1280, 1))
            classifier.append(Hswish(True))
            classifier.append(nn.Conv2d(1280, nclass, 1))
        else:
            raise ValueError('Unknown mode.')
        self.classifier = nn.Sequential(*classifier)

        self._init_weights()

    def _make_layer(self, block, input_channels, block_setting, width_mult, dilation=1, norm_layer=nn.BatchNorm2d):
        layers = list()
        for k, exp_size, c, se, nl, s in block_setting:
            s = 1 if dilation != 1 else s
            out_channels = int(c * width_mult)
            exp_channels = int(exp_size * width_mult)
            layers.append(block(input_channels, out_channels, exp_channels, k, s, dilation, se, nl, norm_layer))
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
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


def get_mobilenet_v3(mode='small', width_mult=1.0, pretrained=False, root='~/,torch/models', **kwargs):
    model = MobileNetV3(mode=mode, width_mult=width_mult, **kwargs)
    if pretrained:
        raise ValueError("Not support pretrained")
    return model


def mobilenet_v3_small_1_0(**kwargs):
    return get_mobilenet_v3('small', 1.0, **kwargs)


if __name__ == '__main__':
    model = mobilenet_v3_small_1_0()



"""Base Model for Semantic Segmentation"""
import torch.nn as nn

__all__ = ['SegBaseModel']


class SegBaseModel(nn.Module):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=True, **kwargs):
        super(SegBaseModel, self).__init__()
        self.nclass = nclass
        self.aux = aux
        self.mode = backbone.split('_')[-1]
        assert self.mode in ['large', 'small']
        if backbone == 'mobilenetv3_small':
            self.pretrained = mobilenet_v3_small_1_0(dilated=True, pretrained=pretrained_base, **kwargs)
        else:
            raise RuntimeError('unknown backbone: {}'.format(backbone))

    def base_forward(self, x):
        """forwarding pre-trained network"""
        x = self.pretrained.conv1(x)

        c1 = self.pretrained.layer1(x)
        c2 = self.pretrained.layer2(c1)
        c3 = self.pretrained.layer3(c2)
        c4 = self.pretrained.layer4(c3)
        c4 = self.pretrained.conv5(c4)

        return c1, c2, c3, c4


if __name__ == '__main__':
    model = SegBaseModel(20, pretrained_base=False)


#start with transfering network architecture code for segmentation without referring to data load

#--------------------------------------

"""MobileNet3 for Semantic Segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNetV3Seg', 'get_mobilenetv3_large_seg', 'get_mobilenetv3_small_seg']


class MobileNetV3Seg(SegBaseModel):
    def __init__(self, nclass, aux=False, backbone='mobilenetv3_small', pretrained_base=False, **kwargs):
        super(MobileNetV3Seg, self).__init__(nclass, aux, backbone, pretrained_base, **kwargs)
        self.head = _SegHead(nclass, self.mode, **kwargs)
        if aux:
            inter_channels = 40 if self.mode == 'small' else 24
            self.auxlayer = nn.Conv2d(inter_channels, nclass, 1)

    def forward(self, x):
        size = x.size()[2:]
        _, c2, _, c4 = self.base_forward(x)
        outputs = list()
        x = self.head(c4)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c2)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)


class _SegHead(nn.Module):
    def __init__(self, nclass, mode='small', norm_layer=nn.BatchNorm2d, **kwargs):
        super(_SegHead, self).__init__()
        in_channels = 960 if mode == 'small' else 576
        self.lr_aspp = _LRASPP(in_channels, norm_layer, **kwargs)
        self.project = nn.Conv2d(128, nclass, 1)

    def forward(self, x):
        x = self.lr_aspp(x)
        return self.project(x)


# TODO: check Lite R-ASPP
class _LRASPP(nn.Module):
    """Lite R-ASPP"""

    def __init__(self, in_channels, norm_layer, **kwargs):
        super(_LRASPP, self).__init__()
        out_channels = 128
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self.b1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),  # check it
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat2 = F.interpolate(feat2, size, mode='bilinear', align_corners=True)
        x = feat1 * feat2  # check it
        return x

if __name__ == '__main__':
    model = MobileNetV3Seg(19)