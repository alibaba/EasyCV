# Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/modeling/backbones/det_resnet_vd.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.registry import BACKBONES
from .det_mobilenet_v3 import Activation


class ConvBNLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        groups=1,
        is_vd_mode=False,
        act=None,
        name=None,
    ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self.act = act
        self._pool2d_avg = nn.AvgPool2d(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self._conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            bias=False)
        if name == 'conv1':
            bn_name = 'bn_' + name
        else:
            bn_name = 'bn' + name[3:]
        self._batch_norm = nn.BatchNorm2d(
            out_channels,
            track_running_stats=True,
        )

        if act is not None:
            self._act = Activation(act_type=act, inplace=True)

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self.act is not None:
            y = self._act(y)
        return y


class BottleneckBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            act='relu',
            name=name + '_branch2a')
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + '_branch2b')
        self.conv2 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels * 4,
            kernel_size=1,
            act=None,
            name=name + '_branch2c')

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels * 4,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + '_branch1')

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = torch.add(short, conv2)
        y = F.relu(y)
        return y


class BasicBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            act='relu',
            name=name + '_branch2a')
        self.conv1 = ConvBNLayer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            act=None,
            name=name + '_branch2b')

        if not shortcut:
            self.short = ConvBNLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name=name + '_branch1')

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = short + conv1
        y = F.relu(y)
        return y


@BACKBONES.register_module()
class OCRDetResNet(nn.Module):

    def __init__(self, in_channels=3, layers=50, **kwargs):
        super(OCRDetResNet, self).__init__()

        self.layers = layers
        supported_layers = [18, 34, 50, 101, 152, 200]
        assert layers in supported_layers, \
            'supported layers are {} but input layer is {}'.format(
                supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        elif layers == 200:
            depth = [3, 12, 48, 3]
        num_channels = [64, 256, 512, 1024
                        ] if layers >= 50 else [64, 64, 128, 256]
        num_filters = [64, 128, 256, 512]

        self.conv1_1 = ConvBNLayer(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            act='relu',
            name='conv1_1')
        self.conv1_2 = ConvBNLayer(
            in_channels=32,
            out_channels=32,
            kernel_size=3,
            stride=1,
            act='relu',
            name='conv1_2')
        self.conv1_3 = ConvBNLayer(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            act='relu',
            name='conv1_3')
        self.pool2d_max = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = nn.ModuleList()
        self.out_channels = []
        if layers >= 50:
            for block in range(len(depth)):
                # block_list = []
                block_list = nn.Sequential()
                shortcut = False
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = 'res' + str(block + 2) + 'a'
                        else:
                            conv_name = 'res' + str(block + 2) + 'b' + str(i)
                    else:
                        conv_name = 'res' + str(block + 2) + chr(97 + i)
                    bottleneck_block = BottleneckBlock(
                        in_channels=num_channels[block]
                        if i == 0 else num_filters[block] * 4,
                        out_channels=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        name=conv_name)

                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i),
                                          bottleneck_block)
                self.out_channels.append(num_filters[block] * 4)
                # self.stages.append(nn.Sequential(*block_list))
                self.stages.append(block_list)
        else:
            for block in range(len(depth)):
                # block_list = []
                block_list = nn.Sequential()
                shortcut = False
                for i in range(depth[block]):
                    conv_name = 'res' + str(block + 2) + chr(97 + i)
                    basic_block = BasicBlock(
                        in_channels=num_channels[block]
                        if i == 0 else num_filters[block],
                        out_channels=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        shortcut=shortcut,
                        if_first=block == i == 0,
                        name=conv_name)

                    shortcut = True
                    block_list.add_module('bb_%d_%d' % (block, i), basic_block)
                    # block_list.append(basic_block)
                self.out_channels.append(num_filters[block])
                self.stages.append(block_list)

                # self.stages.append(nn.Sequential(*block_list))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        out = []
        for block in self.stages:
            y = block(y)
            out.append(y)
        return out
