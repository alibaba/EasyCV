# Copyright (c) Alibaba, Inc. and its affiliates.
r""" This model is taken from the official PyTorch model zoo.
     - torchvision.models.mobilenet.py on 31th Aug, 2019
"""

from torch import nn

from easycv.framework.errors import ValueError
from ..modelzoo import mobilenetv2 as model_urls
from ..registry import BACKBONES

__all__ = ['MobileNetV2']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False), nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True))


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(
                hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


@BACKBONES.register_module
class MobileNetV2(nn.Module):

    def __init__(self,
                 num_classes=0,
                 width_multi=1.0,
                 inverted_residual_setting=None,
                 round_nearest=8):
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_multi (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
        """
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError('inverted_residual_setting should be non-empty '
                             'or a 4-element list, got {}'.format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_multi,
                                        round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_multi), round_nearest)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_multi, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(
            ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(self.last_channel, num_classes),
            )
        self.default_pretrained_model_path = model_urls[self.__class__.__name__
                                                        + '_' +
                                                        str(width_multi)]

    def init_weights(self):
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
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        if hasattr(self, 'classifier'):
            x = x.mean([2, 3])
            x = self.classifier(x)
        return [x]
