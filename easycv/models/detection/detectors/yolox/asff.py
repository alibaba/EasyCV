# Copyright (c) 2014-2021 Alibaba PAI-Teams and GOATmessi7. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.framework.errors import ValueError
from easycv.models.backbones.network_blocks import BaseConv


class ASFF(nn.Module):

    def __init__(self,
                 level,
                 type='ASFF',
                 asff_channel=2,
                 expand_kernel=3,
                 multiplier=1,
                 act='silu'):
        """
        Args:
            level(int): the level of the input feature
            type(str): ASFF or ASFF_sim
            asff_channel(int): the hidden channel of the attention layer in ASFF
            expand_kernel(int): expand kernel size of the expand layer
            multiplier: should be the same as width in the backbone
        """
        super(ASFF, self).__init__()
        self.level = level
        self.type = type

        self.dim = [
            int(1024 * multiplier),
            int(512 * multiplier),
            int(256 * multiplier)
        ]

        Conv = BaseConv

        self.inter_dim = self.dim[self.level]

        if self.type == 'ASFF':
            if level == 0:
                self.stride_level_1 = Conv(
                    int(512 * multiplier), self.inter_dim, 3, 2, act=act)

                self.stride_level_2 = Conv(
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            elif level == 1:
                self.compress_level_0 = Conv(
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                self.stride_level_2 = Conv(
                    int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            elif level == 2:
                self.compress_level_0 = Conv(
                    int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
                self.compress_level_1 = Conv(
                    int(512 * multiplier), self.inter_dim, 1, 1, act=act)
            else:
                raise ValueError('Invalid level {}'.format(level))

        # add expand layer
        self.expand = Conv(
            self.inter_dim, self.inter_dim, expand_kernel, 1, act=act)

        self.weight_level_0 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_1 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)
        self.weight_level_2 = Conv(self.inter_dim, asff_channel, 1, 1, act=act)

        self.weight_levels = Conv(asff_channel * 3, 3, 1, 1, act=act)

    def expand_channel(self, x):
        # [b,c,h,w]->[b,c*4,h/2,w/2]
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (
                patch_top_left,
                patch_bot_left,
                patch_top_right,
                patch_bot_right,
            ),
            dim=1,
        )
        return x

    def mean_channel(self, x):
        # [b,c,h,w]->[b,c/4,h*2,w*2]
        x1 = x[:, ::2, :, :]
        x2 = x[:, 1::2, :, :]
        return (x1 + x2) / 2

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[2]  # max feature level [512,20,20]
        x_level_1 = x[1]  # mid feature level [256,40,40]
        x_level_2 = x[0]  # min feature level [128,80,80]

        if self.type == 'ASFF':
            if self.level == 0:
                level_0_resized = x_level_0
                level_1_resized = self.stride_level_1(x_level_1)
                level_2_downsampled_inter = F.max_pool2d(
                    x_level_2, 3, stride=2, padding=1)
                level_2_resized = self.stride_level_2(
                    level_2_downsampled_inter)
            elif self.level == 1:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = F.interpolate(
                    level_0_compressed, scale_factor=2, mode='nearest')
                level_1_resized = x_level_1
                level_2_resized = self.stride_level_2(x_level_2)
            elif self.level == 2:
                level_0_compressed = self.compress_level_0(x_level_0)
                level_0_resized = F.interpolate(
                    level_0_compressed, scale_factor=4, mode='nearest')
                x_level_1_compressed = self.compress_level_1(x_level_1)
                level_1_resized = F.interpolate(
                    x_level_1_compressed, scale_factor=2, mode='nearest')
                level_2_resized = x_level_2
        else:
            if self.level == 0:
                level_0_resized = x_level_0
                level_1_resized = self.expand_channel(x_level_1)
                level_1_resized = self.mean_channel(level_1_resized)
                level_2_resized = self.expand_channel(x_level_2)
                level_2_resized = F.max_pool2d(
                    level_2_resized, 3, stride=2, padding=1)
            elif self.level == 1:
                level_0_resized = F.interpolate(
                    x_level_0, scale_factor=2, mode='nearest')
                level_0_resized = self.mean_channel(level_0_resized)
                level_1_resized = x_level_1
                level_2_resized = self.expand_channel(x_level_2)
                level_2_resized = self.mean_channel(level_2_resized)

            elif self.level == 2:
                level_0_resized = F.interpolate(
                    x_level_0, scale_factor=4, mode='nearest')
                level_0_resized = self.mean_channel(
                    self.mean_channel(level_0_resized))
                level_1_resized = F.interpolate(
                    x_level_1, scale_factor=2, mode='nearest')
                level_1_resized = self.mean_channel(level_1_resized)
                level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat(
            (level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:
                                                            1, :, :] + level_1_resized * levels_weight[:,
                                                                                                       1:
                                                                                                       2, :, :] + level_2_resized * levels_weight[:,
                                                                                                                                                  2:, :, :]
        out = self.expand(fused_out_reduced)

        return out
