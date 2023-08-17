import torch
import torch.nn as nn
import torch.nn.functional as F

class FireModule(nn.Module):
    def __init__(self, fire_i, base_e, freq, squeeze_ratio, pct_3x3, dilation_rate,
                 activation, kernel_initializer, data_format, use_bias=False, decoder=False):
        super(FireModule, self).__init__()
        e_i, s_1x1, e_1x1, e_3x3 = self.get_fire_config(
            fire_i, base_e, freq, squeeze_ratio, pct_3x3)
        self.decoder = decoder

        if decoder:
            d = "decoder_"
        else:
            d = ""

        self.squeeze = nn.Conv2d(
            in_channels=e_i, out_channels=s_1x1, kernel_size=1, bias=use_bias)
        self.fire2_expand1 = nn.Conv2d(
            in_channels=s_1x1, out_channels=e_1x1, kernel_size=1, bias=use_bias)
        self.fire2_expand2 = nn.Conv2d(
            in_channels=s_1x1, out_channels=e_3x3, kernel_size=3, padding=1, dilation=dilation_rate, bias=use_bias)
        
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
    def __init__(self, nb_classes=2, base_e=64, freq=4, squeeze_ratio=0.25, pct_3x3=0.5,
                 inputs_shape=(3, 224, 224), use_bias=False, data_format="channels_first",
                 activation=nn.ReLU(), kernel_initializer=None):
        super(MicroNet, self).__init__()

        self.inputs = nn.Conv2d(inputs_shape[0], inputs_shape[0], kernel_size=1)
        
        # Encoder
        self.conv1 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format)
        self.conv2 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format)
        self.conv3 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 2, activation, kernel_initializer, data_format)
        self.conv4 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 3, activation, kernel_initializer, data_format)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format)
        self.conv6 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format)
        self.conv7 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 2, activation, kernel_initializer, data_format)
        self.conv8 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 3, activation, kernel_initializer, data_format)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv9 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format)
        self.conv10 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format)
        self.conv11 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 2, activation, kernel_initializer, data_format)
        self.conv12 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 3, activation, kernel_initializer, data_format)

        # Decoder
        self.d_conv11 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 3, activation, kernel_initializer, data_format, decoder=True)
        self.d_conv10 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 2, activation, kernel_initializer, data_format, decoder=True)
        self.d_conv9 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format, decoder=True)
        self.up2 = nn.ConvTranspose2d(
            in_channels=128, out_channels=128, kernel_size=2, stride=2)

        self.d_conv8 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 3, activation, kernel_initializer, data_format, decoder=True)
        self.d_conv7 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 2, activation, kernel_initializer, data_format, decoder=True)
        self.d_conv6 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format, decoder=True)
        self.up1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=2, stride=2)

        self.d_conv5 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 3, activation, kernel_initializer, data_format, decoder=True)
        self.d_conv4 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 2, activation, kernel_initializer, data_format, decoder=True)
        self.d_conv3 = self.create_fire_modules(
            base_e, freq, squeeze_ratio, pct_3x3, 1, activation, kernel_initializer, data_format, decoder=True)

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
        d_conv11 = self.d_conv11(conv11)
        d_conv10 = self.d_conv10(d_conv11)
        d_conv9 = self.d_conv9(d_conv10)
        up2 = self.up2(d_conv9)

        d_conv8 = self.d_conv8(up2 + conv8)
        d_conv7 = self.d_conv7(d_conv8)
        d_conv6 = self.d_conv6(d_conv7)
        up1 = self.up1(d_conv6)

        d_conv5 = self.d_conv5(up1 + conv5)
        d_conv4 = self.d_conv4(d_conv5)
        d_conv3 = self.d_conv3(d_conv4)

        # Classifier
        out_conv = self.out_conv(d_conv3)
        return F.softmax(out_conv, dim=1)

    def create_fire_modules(self, base_e, freq, squeeze_ratio, pct_3x3, dilation_rate,
                            activation, kernel_initializer, data_format, use_bias=False, decoder=False):
        return FireModule(0, base_e, freq, squeeze_ratio, pct_3x3, dilation_rate,
                          activation, kernel_initializer, data_format, use_bias, decoder)

# Instantiate the MicroNet model
model = MicroNet(nb_classes=2)

# Print the model's architecture
print(model)
