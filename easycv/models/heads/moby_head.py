# Copyright (c) Alibaba, Inc. and its affiliates.
#
# Used for self-supervised learning MoBY algorithm, when export_head=True in the config file
#
# Author: YANG Ruixin
# GitHub: https://github.com/yang-ruixin
# Email: yang_ruixin@126.com
# Date: 2023/01/05

# from typing import Dict, List

# import torch
import torch.nn as nn
# from mmcv.cnn.utils.weight_init import initialize
#
# from easycv.core.evaluation.metrics import accuracy
# from easycv.utils.logger import get_root_logger
# from easycv.utils.registry import build_from_cfg
from ..registry import HEADS  # , LOSSES

from ..utils import _init_weights


@HEADS.register_module
class MoBYMLP(nn.Module):

    def __init__(self,
                 in_channels=256,
                 hid_channels=4096,
                 out_channels=256,
                 num_layers=2,
                 with_avg_pool=True):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(
                nn.Linear(in_channels if i == 0 else hid_channels,
                          hid_channels))
            linear_hidden.append(nn.BatchNorm1d(hid_channels))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)
        self.linear_out = nn.Linear(
            in_channels if num_layers == 1 else hid_channels,
            out_channels) if num_layers >= 1 else nn.Identity()
        self.with_avg_pool = True
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x[0]
        if self.with_avg_pool and len(x.shape) == 4:
            bs = x.shape[0]
            x = self.avg_pool(x).view([bs, -1])
        # print(x.shape)
        # exit()
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return [x]

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)
