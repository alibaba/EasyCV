# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn


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
