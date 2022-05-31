# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from easycv.utils.registry import Registry, build_from_cfg

ACTIVATION_LAYERS = Registry('activation layer')

for module in [
        nn.ReLU, nn.LeakyReLU, nn.PReLU, nn.RReLU, nn.ReLU6, nn.ELU,
        nn.Sigmoid, nn.Tanh
]:
    ACTIVATION_LAYERS.register_module(module)


@ACTIVATION_LAYERS.register_module()
class FReLU(nn.Module):

    def __init__(self, in_channel):
        super().__init__()
        self.depthwise_conv_bn = nn.Sequential(
            nn.Conv2d(
                in_channel,
                in_channel,
                3,
                padding=1,
                groups=in_channel,
                bias=False), nn.BatchNorm2d(in_channel))

    def forward(self, x):
        funnel_x = self.depthwise_conv_bn(x)
        return torch.max(x, funnel_x)


def build_activation_layer(cfg):
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return build_from_cfg(cfg, ACTIVATION_LAYERS)
