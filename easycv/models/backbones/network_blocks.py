# Copyright (c) 2014-2021 Megvii Inc, AlanLi And Alibaba PAI Team. All rights reserved.

import torch
import torch.nn as nn


class SiLU(nn.Module):
    """export-friendly inplace version of nn.SiLU()"""

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.sigmoid_(x)
        return x * torch.sigmoid(x)


class HSiLU(nn.Module):
    """
        export-friendly inplace version of nn.SiLU()
        hardsigmoid is better than sigmoid when used for edge model
    """

    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace

    @staticmethod
    def forward(x):
        # clone is not supported with nni 2.6.1
        # result = x.clone()
        # torch.hardsigmoid(x)
        return x * torch.hardsigmoid(x)


def get_activation(name='silu', inplace=True):
    if name == 'silu':
        # @ to do nn.SiLU 1.7.0
        # module = nn.SiLU(inplace=inplace)
        module = SiLU(inplace=inplace)
    elif name == 'relu':
        module = nn.ReLU(inplace=inplace)
    elif name == 'lrelu':
        module = nn.LeakyReLU(0.1, inplace=inplace)
    elif name == 'hsilu':
        module = HSiLU(inplace=inplace)
    elif name == 'identity':
        module = nn.Identity(inplace=inplace)
    else:
        raise AttributeError('Unsupported act type: {}'.format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act='silu'):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act='silu'):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act)

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act='silu',
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class ResLayer(nn.Module):
    'Residual layer with `in_channels` inputs.'

    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.layer1 = BaseConv(
            in_channels, mid_channels, ksize=1, stride=1, act='lrelu')
        self.layer2 = BaseConv(
            mid_channels, in_channels, ksize=3, stride=1, act='lrelu')

    def forward(self, x):
        out = self.layer2(self.layer1(x))
        return x + out


class SPPFBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=5,
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=activation)

        self.m = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

        conv2_channels = hidden_channels * 4
        self.conv2 = BaseConv(
            conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.m(x)
        x2 = self.m(x1)
        x = self.conv2(torch.cat([x, x1, x2, self.m(x2)], 1))

        return x


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=activation)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(
            conv2_channels, out_channels, 1, stride=1, act=activation)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPLayer(nn.Module):
    """CSP Bottleneck with 3 convolutions"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n=1,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act='silu',
    ):
        """
        Args:
            in_channels (int): input channels.
            out_channels (int): output channels.
            n (int): number of Bottlenecks. Default value: 1.
        """
        # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        hidden_channels = int(out_channels * expansion)  # hidden channels
        self.conv1 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(
            in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(
            2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(
                hidden_channels,
                hidden_channels,
                shortcut,
                1.0,
                depthwise,
                act=act) for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=1,
                 stride=1,
                 act='silu'):
        super().__init__()
        self.conv = BaseConv(
            in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        # shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
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
        return self.conv(x)


class GSConv(nn.Module):
    """
        GSConv is used to merge the channel information of DSConv and BaseConv
        You can refer to https://github.com/AlanLi1997/slim-neck-by-gsconv for more details
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act='silu'):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = BaseConv(c1, c_, k, s, g, act)
        self.cv2 = BaseConv(c_, c_, 5, 1, c_, act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)


class GSBottleneck(nn.Module):
    """
        The use of GSBottleneck is to stack the GSConv layer
        You can refer to https://github.com/AlanLi1997/slim-neck-by-gsconv for more details
    """

    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2

        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1), GSConv(c_, c2, 1, 1, act='identity'))

    def forward(self, x):
        return self.conv_lighting(x)


class VoVGSCSP(nn.Module):
    """
        VoVGSCSP is a new neck structure used in CSPNet
        You can refer to https://github.com/AlanLi1997/slim-neck-by-gsconv for more details
    """

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = BaseConv(c1, c_, 1, 1)
        self.cv2 = BaseConv(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(*(GSBottleneck(c_, c_) for _ in range(n)))

    def forward(self, x):
        x1 = self.cv1(x)
        return self.cv2(torch.cat((self.m(x1), x1), dim=1))
