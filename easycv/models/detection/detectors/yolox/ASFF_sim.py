import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.backbones.network_blocks import DWConv, SiLU


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        # @ to do nn.SiLU 1.7.0
        # module = nn.SiLU(inplace=inplace)
        module = SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class Conv(nn.Module):
    # Standard convolution
    def __init__(self,
                 c1,
                 c2,
                 k=1,
                 s=1,
                 p=None,
                 g=1,
                 act='silu'):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ASFF(nn.Module):

    def __init__(self,
                 level,
                 multiplier=1,
                 asff_channel=2,
                 expand_kernel=3,
                 down_rate=None,
                 use_dconv=False,
                 use_expand=True,
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

        self.inter_dim = self.dim[self.level]

        self.use_expand = use_expand

        if level == 0:
            if down_rate == None:
                self.expand = Conv(
                    self.inter_dim,
                    int(1024 * multiplier),
                    expand_kernel,
                    1,
                    act=act)
            else:
                if use_dconv:
                    self.expand = DWConv(
                        self.inter_dim,
                        int(1024 * multiplier),
                        expand_kernel,
                        1,
                        act=act)
                else:
                    self.expand = nn.Sequential(
                        Conv(
                            self.inter_dim,
                            int(self.inter_dim // down_rate),
                            1,
                            1,
                            act=act),
                        Conv(
                            int(self.inter_dim // down_rate),
                            int(1024 * multiplier),
                            1,
                            1,
                            act=act))

        elif level == 1:
            if down_rate == None:
                self.expand = Conv(
                    self.inter_dim,
                    int(512 * multiplier),
                    expand_kernel,
                    1,
                    act=act)
            else:
                if use_dconv:
                    self.expand = DWConv(
                        self.inter_dim,
                        int(512 * multiplier),
                        expand_kernel,
                        1,
                        act=act)
                else:
                    self.expand = nn.Sequential(
                        Conv(
                            self.inter_dim,
                            int(self.inter_dim // down_rate),
                            1,
                            1,
                            act=act),
                        Conv(
                            int(self.inter_dim // down_rate),
                            int(512 * multiplier),
                            1,
                            1,
                            act=act))

        elif level == 2:
            if down_rate == None:
                self.expand = Conv(
                    self.inter_dim,
                    int(256 * multiplier),
                    expand_kernel,
                    1,
                    act=act)
            else:
                if use_dconv:
                    self.expand = DWConv(
                        self.inter_dim,
                        int(256 * multiplier),
                        expand_kernel,
                        1,
                        act=act)
                else:
                    self.expand = nn.Sequential(
                        Conv(
                            self.inter_dim,
                            int(self.inter_dim // down_rate),
                            1,
                            1,
                            act=act),
                        Conv(
                            int(self.inter_dim // down_rate),
                            int(256 * multiplier),
                            1,
                            1,
                            act=act))

        # when adding rfb, we use half number of channels to save memory
        # compress_c = 8 if rfb else 16
        compress_c = asff_channel

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1, act=act)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1, act=act)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1, act=act)

        self.weight_levels = Conv(compress_c * 3, 3, 1, 1, act=act)
        self.vis = vis

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
        x_level_0 = x[2]  # max feature [512,20,20]
        x_level_1 = x[1]  # mid feature [256,40,40]
        x_level_2 = x[0]  # min feature [128,80,80]

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

        if self.use_expand:
            out = self.expand(fused_out_reduced)
        else:
            out = fused_out_reduced

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


# if __name__ == '__main__':
#     width = 0.5
#     num_classes = 80
#     in_channels = [256, 512, 1024]
#
#     asff_channel = 2
#     act = 'relu'
#
#     asff_1 = ASFF(
#         level=0, multiplier=width, asff_channel=asff_channel, act=act).cuda()
#     asff_2 = ASFF(
#         level=1, multiplier=width, asff_channel=asff_channel, act=act).cuda()
#     asff_3 = ASFF(
#         level=2, multiplier=width, asff_channel=asff_channel, act=act).cuda()
#
#     input = (torch.rand(1, 128, 80, 80).cuda(), torch.rand(1, 256, 40,
#                                                            40).cuda(),
#              torch.rand(1, 512, 20, 20).cuda())
#
#     # flops, params = get_model_complexity_info(asff_1, input, as_strings=True,
#     #                                           print_per_layer_stat=True)
#     # print('Flops:  ' + flops)
#     # print('Params: ' + params)
#
#     # input = torch.randn(1, 3, 640, 640).cuda()
#     # flops, params = profile(asff_1, inputs=(input,))
#     # print('flops: {}, params: {}'.format(flops, params))
#
#     from torchsummaryX import summary
#
#     summary(asff_1, input)
#     summary(asff_2, input)
#     summary(asff_3, input)
