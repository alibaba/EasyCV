# borrow some code from https://github.com/DingXiaoH/RepVGG/repvgg.py MIT2.0
import warnings

import numpy as np
import torch
import torch.nn as nn

from easycv.models.utils.ops import make_divisible


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module(
        'conv',
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """
        Basic Block of RepVGG
        It's an efficient block that will be reparameterized in evaluation. (deploy = True)
        Usage: RepVGGBlock(in_channels, out_channels, ksize=3, stride=stride)

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize=3,
                 stride=1,
                 padding=1,
                 dilation=1,
                 groups=1,
                 padding_mode='zeros',
                 deploy=False,
                 act=None):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert ksize == 3
        assert padding == 1

        padding_11 = padding - ksize // 2

        self.nonlinearity = nn.ReLU()
        self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(
                num_features=in_channels
            ) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ksize,
                stride=stride,
                padding=padding,
                groups=groups)
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups)

    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))

    #   Optional. This improves the accuracy and facilitates quantization.
    #   1.  Cancel the original weight decay on rbr_dense.conv.weight and rbr_1x1.conv.weight.
    #   2.  Use like this.
    #       loss = criterion(....)
    #       for every RepVGGBlock blk:
    #           loss += weight_decay_coefficient * 0.5 * blk.get_cust_L2()
    #       optimizer.zero_grad()
    #       loss.backward()

    def get_custom_L2(self):
        K3 = self.rbr_dense.conv.weight
        K1 = self.rbr_1x1.conv.weight
        t3 = (self.rbr_dense.bn.weight /
              ((self.rbr_dense.bn.running_var +
                self.rbr_dense.bn.eps).sqrt())).reshape(-1, 1, 1, 1).detach()
        t1 = (self.rbr_1x1.bn.weight / ((self.rbr_1x1.bn.running_var +
                                         self.rbr_1x1.bn.eps).sqrt())).reshape(
                                             -1, 1, 1, 1).detach()

        l2_loss_circle = (K3**2).sum() - (K3[:, :, 1:2, 1:2]**2).sum(
        )  # The L2 loss of the "circle" of weights in 3x3 kernel. Use regular L2 on them.
        eq_kernel = K3[:, :, 1:2, 1:
                       2] * t3 + K1 * t1  # The equivalent resultant central point of 3x3 kernel.
        l2_loss_eq_kernel = (eq_kernel**2 / (t3**2 + t1**2)).sum(
        )  # Normalize for an L2 coefficient comparable to regular L2.
        return l2_loss_eq_kernel + l2_loss_circle

    #   This func derives the equivalent kernel and bias in a DIFFERENTIABLE way.
    #   You can get the equivalent kernel and bias at any time and do whatever you want,
    #   for example, apply some penalties or constraints during training, just like you do to the other models.
    #   May be useful for quantization or pruning.
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3),
                                        dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(
                    branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True


class ConvBNAct(nn.Module):
    '''Normal Conv with SiLU activation'''

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False,
                 act='relu'):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if act == 'relu':
            self.act = nn.ReLU()
        if act == 'silu':
            self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class ConvBNReLU(ConvBNAct):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            act='relu')


class ConvBNSiLU(ConvBNAct):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            bias=bias,
            act='silu')


class MT_SPPF(nn.Module):
    '''Simplified SPPF with ReLU activation'''

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = ConvBNReLU(in_channels, c_, 1, 1)
        self.cv2 = ConvBNReLU(c_ * 4, out_channels, 1, 1)
        self.maxpool = nn.MaxPool2d(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.maxpool(x)
            y2 = self.maxpool(y1)
            return self.cv2(torch.cat([x, y1, y2, self.maxpool(y2)], 1))


class RepVGGYOLOX(nn.Module):
    '''
        RepVGG with MT_SPPF to build a efficient Yolox backbone
    '''

    def __init__(
        self,
        in_channels=3,
        depth=1.0,
        width=1.0,
    ):
        super().__init__()
        num_repeat_backbone = [1, 6, 12, 18, 6]
        channels_list_backbone = [64, 128, 256, 512, 1024]
        num_repeat_neck = [12, 12, 12, 12]
        channels_list_neck = [256, 128, 128, 256, 256, 512]
        num_repeats = [(max(round(i * depth), 1) if i > 1 else i)
                       for i in (num_repeat_backbone + num_repeat_neck)]

        channels_list = [
            make_divisible(i * width, 8)
            for i in (channels_list_backbone + channels_list_neck)
        ]

        assert channels_list is not None
        assert num_repeats is not None

        self.stage0 = RepVGGBlock(
            in_channels=in_channels,
            out_channels=channels_list[0],
            ksize=3,
            stride=2)
        self.stage1 = self._make_stage(channels_list[0], channels_list[1],
                                       num_repeats[1])
        self.stage2 = self._make_stage(channels_list[1], channels_list[2],
                                       num_repeats[2])
        self.stage3 = self._make_stage(channels_list[2], channels_list[3],
                                       num_repeats[3])
        self.stage4 = self._make_stage(
            channels_list[3], channels_list[4], num_repeats[4], add_ppf=True)

    def _make_stage(self,
                    in_channels,
                    out_channels,
                    repeat,
                    stride=2,
                    add_ppf=False):
        blocks = []
        blocks.append(
            RepVGGBlock(in_channels, out_channels, ksize=3, stride=stride))
        for i in range(repeat):
            blocks.append(RepVGGBlock(out_channels, out_channels))
        if add_ppf:
            blocks.append(MT_SPPF(out_channels, out_channels, kernel_size=5))

        return nn.Sequential(*blocks)

    def forward(self, x):
        outputs = []
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        outputs.append(x)
        x = self.stage3(x)
        outputs.append(x)
        x = self.stage4(x)
        outputs.append(x)
        return tuple(outputs)
