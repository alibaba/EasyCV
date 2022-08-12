import torch
import torch.nn as nn
import torch.nn.functional as F
from easycv.models.backbones.network_blocks import BaseConv, DWConv, SiLU

class ASFF(nn.Module):

    def __init__(self,
                 level,
                 multiplier=1,
                 asff_channel=16,
                 rfb=False,
                 vis=False,
                 act='silu'):
        """
        multiplier should be 1, 0.5
        which means, the channel of ASFF can be
        512, 256, 128 -> multiplier=0.5
        1024, 512, 256 -> multiplier=1
        For even smaller, you need change code manually.
        """
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [
            int(1024 * multiplier),
            int(512 * multiplier),
            int(256 * multiplier)
        ]

        Conv = BaseConv

        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 3, 2, act=act)

            self.stride_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 3, 2, act=act)

            self.expand = Conv(
                self.inter_dim, int(1024 * multiplier), 3, 1, act=act)
        elif level == 1:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
            self.stride_level_2 = Conv(
                int(256 * multiplier), self.inter_dim, 3, 2, act=act)
            self.expand = Conv(
                self.inter_dim, int(512 * multiplier), 3, 1, act=act)
        elif level == 2:
            self.compress_level_0 = Conv(
                int(1024 * multiplier), self.inter_dim, 1, 1, act=act)
            self.compress_level_1 = Conv(
                int(512 * multiplier), self.inter_dim, 1, 1, act=act)
            self.expand = Conv(
                self.inter_dim, int(256 * multiplier), 3, 1, act=act)

        # when adding rfb, we use half number of channels to save memory
        # compress_c = 8 if rfb else 16
        compress_c = asff_channel

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1, act=act)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1, act=act)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1, act=act)

        self.weight_levels = Conv(compress_c * 3, 3, 1, 1, act=act)
        self.vis = vis

    def forward(self, x):  # l,m,s
        """
        #
        256, 512, 1024
        from small -> large
        """
        x_level_0 = x[2]  # max feature level [512,20,20]
        x_level_1 = x[1]  # mid feature level [256,40,40]
        x_level_2 = x[0]  # min feature level [128,80,80]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(
                x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
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

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
