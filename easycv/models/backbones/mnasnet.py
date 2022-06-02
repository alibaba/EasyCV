# Copyright (c) Alibaba, Inc. and its affiliates.
r""" This model is taken from the official PyTorch model zoo.
     - torchvision.models.mnasnet.py on 31th Aug, 2019
"""

import torch
from torch import nn

from ..modelzoo import mnasnet as model_urls
from ..registry import BACKBONES

__all__ = ['MNASNet']

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 kernel_size,
                 stride,
                 expansion_factor,
                 bn_momentum=0.1):
        super(_InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(
                mid_ch,
                mid_ch,
                kernel_size,
                padding=kernel_size // 2,
                stride=stride,
                groups=mid_ch,
                bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats,
           bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(
        in_ch,
        out_ch,
        kernel_size,
        stride,
        exp_factor,
        bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(
                out_ch,
                out_ch,
                kernel_size,
                1,
                exp_factor,
                bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _scale_depths(depths, alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


@BACKBONES.register_module
class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """

    def __init__(self, alpha, num_classes=0, dropout=0.2):
        super(MNASNet, self).__init__()
        depths = _scale_depths([24, 40, 80, 96, 192, 320], alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(16, depths[0], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[0], depths[1], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[1], depths[2], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[5], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)

        if num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, num_classes))

        self.default_pretrained_model_path = model_urls[self.__class__.__name__
                                                        + str(alpha)]

    def forward(self, x):
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        if hasattr(self, 'classifier'):
            x = x.mean([2, 3])
            return [self.classifier(x)]
        else:
            return [x]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)
