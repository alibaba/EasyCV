# Copyright (c) Alibaba, Inc. and its affiliates.
import uuid

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..modelzoo import genet as model_urls
from ..registry import BACKBONES

GENET_LARGE = 'ConvKX(uuid9d1dca0f098143aaa1a947acf1100787|3,32,3,2)\
BN(uuid7d10ba10dc524ffb8863ae97c4a21797|32)RELU(uuidccd810d3d10a48158ccfa48ca975915c|32)\
SuperResKXKX(uuid5ba1db21fce64b16a34ad577c258fd6c|32,128,3,2,1.0,1)\
SuperResKXKX(uuida09fc4e4946444bf9b912f8c666c4b12|128,192,3,2,1.0,2)\
SuperResK1KXK1(uuidfa45c5f5cc96435dbd54801f31c83ca8|192,640,3,2,0.25,6)\
SuperResK1DWK1(uuid99bf6442b33643579dc680045da7549d|640,640,3,2,3.0,5)\
SuperResK1DWK1(uuid615cbfd4ed284cbc8589d84cbe9b0e92|640,640,3,1,3.0,4)\
ConvKX(uuid002fa25f74f14cdeb89a5aacd6ce64ff|640,2560,1,1)\
BN(uuidc5d6c88c326343efa2a8700907f87732|2560)RELU(uuidd2b39caab4cb4ac2b6905b18858c0037|2560)AdaptiveAvgPool(2560,1)'

GENET_NORMAL = 'ConvKX(uuid70de938099844017bd745349f7a1d35a|3,32,3,2)\
BN(uuid10f8a99f83294067bfdf5fc5a5c9bffd|32)\
RELU(uuideffe03bd73254e7c8027364ba71d25cd|32)\
SuperResKXKX(uuidb023bea8c7b34c22a1650e07dfc8e2c1|32,128,3,2,1.0,1)\
SuperResKXKX(uuidf829740023044b879eefaf7fc7d1ad8e|128,192,3,2,1.0,2)\
SuperResK1KXK1(uuid33bfe77cb8864357a840ca3341ea629a|192,640,3,2,0.25,6)\
SuperResK1DWK1(uuide2c948d819fb4869980e30d67a773244|640,640,3,2,3.0,4)\
SuperResK1DWK1(uuid53c308e481c24154b7a81fcbaf99edbf|640,640,3,1,3.0,1)\
ConvKX(uuidbc6953bfd8de45fc8534787a66b96430|640,2560,1,1)\
BN(uuida8acaaae74ed47a4a7514b41c643eb23|2560)RELU(uuida5d71c4fd5d24a7b848472f0383df467|2560)AdaptiveAvgPool(2560,1)'

GENET_SMALL = 'ConvKX(uuid46ff2328b77f40ff88aed69a5318d771|3,13,3,2)\
BN(uuid43b72f65311c42d9a1af485c594a6ab4|13)RELU(uuid282901aaa7f84b028e3c5bd7d37ae056|13)\
SuperResKXKX(uuiddb56d6f9a60b4455966e13b06a8ff723|13,48,3,2,1.0,1)\
SuperResKXKX(uuidd964406e6fdf4e9abac225afaeb1fe0b|48,48,3,2,1.0,3)\
SuperResK1KXK1(uuid39819ad4f4da405583de614af437b568|48,384,3,2,0.25,7)\
SuperResK1DWK1(uuid420593fe7b1e46f690b76bac3786d4b7|384,560,3,2,3.0,2)\
SuperResK1DWK1(uuid96236b3c50774f1ab2d3049d6aca6d85|560,256,3,1,3.0,1)\
ConvKX(uuid89ed263767a14f21b7426cccb120ad1d|256,1920,1,1)\
BN(uuidd6ad568b290544be9f4b47dc3fa271c9|1920)RELU(uuid823ced7441394fb9b3a96a5f7c40da2b|1920)AdaptiveAvgPool(1920,1)'

plainnet_struct_dict = {
    'normal': GENET_NORMAL,
    'large': GENET_LARGE,
    'small': GENET_SMALL
}

# ------------ Fuse BN ------


def _fuse_convkx_and_bn_(convkx, bn):
    the_weight_scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
    convkx.weight[:] = convkx.weight * the_weight_scale.view((-1, 1, 1, 1))
    the_bias_shift = (bn.weight * bn.running_mean) / \
        torch.sqrt(bn.running_var + bn.eps)
    bn.weight[:] = 1
    bn.bias[:] = bn.bias - the_bias_shift
    bn.running_var[:] = 1.0 - bn.eps
    bn.running_mean[:] = 0.0
    convkx.bias = nn.Parameter(bn.bias)


def remove_bn_in_superblock(super_block):

    new_shortcut_list = []
    for the_seq_list in super_block.shortcut_list:
        assert isinstance(the_seq_list, nn.Sequential)
        new_seq_list = []
        last_block = None
        for block in the_seq_list:
            if isinstance(block, nn.BatchNorm2d):
                _fuse_convkx_and_bn_(last_block, block)
            else:
                new_seq_list.append(block)
            last_block = block
        new_shortcut_list.append(nn.Sequential(*new_seq_list))

    super_block.shortcut_list = nn.ModuleList(new_shortcut_list)

    new_conv_list = []
    for the_seq_list in super_block.conv_list:
        assert isinstance(the_seq_list, nn.Sequential)
        new_seq_list = []
        last_block = None
        for block in the_seq_list:
            if isinstance(block, nn.BatchNorm2d):
                _fuse_convkx_and_bn_(last_block, block)
            else:
                new_seq_list.append(block)
            last_block = block
        new_conv_list.append(nn.Sequential(*new_seq_list))

    super_block.conv_list = nn.ModuleList(new_conv_list)


def fuse_bn(model):
    the_block_list = model.block_list
    last_block = the_block_list[0]
    new_block_list = [last_block]
    for the_block in the_block_list[1:]:
        if isinstance(the_block, BN):
            _fuse_convkx_and_bn_(last_block.netblock, the_block.netblock)
        else:
            new_block_list.append(the_block)
        last_block = the_block
    pass

    the_block_list = new_block_list
    for the_block in the_block_list:
        if hasattr(the_block, 'shortcut_list'):
            remove_bn_in_superblock(the_block)
        else:
            continue

    model.block_list = new_block_list
    model.module_list = nn.ModuleList(new_block_list)

    return model


# ------------ end of fuse bn --------


def _create_netblock_list_from_str_(s, no_create=False):
    block_list = []
    while len(s) > 0:
        is_found_block_class = False
        for the_block_class_name in _all_netblocks_dict_.keys():
            if s.startswith(the_block_class_name):
                is_found_block_class = True
                the_block_class = _all_netblocks_dict_[the_block_class_name]
                the_block, remaining_s = the_block_class.create_from_str(
                    s, no_create=no_create)
                if the_block is not None:
                    block_list.append(the_block)
                s = remaining_s
                if len(s) > 0 and s[0] == ';':
                    return block_list, s[1:]
                break
            pass  # end if
        pass  # end for
        assert is_found_block_class
    pass  # end while
    return block_list, ''


def _get_right_parentheses_index_(s):
    # assert s[0] == '('
    left_paren_count = 0
    for index, x in enumerate(s):

        if x == '(':
            left_paren_count += 1
        elif x == ')':
            left_paren_count -= 1
            if left_paren_count == 0:
                return index
        else:
            pass
    return None


'''
-------------------- GENet Blocks --------------------
'''


class PlainNetBasicBlockClass(nn.Module):

    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 stride=1,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(PlainNetBasicBlockClass, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.no_create = no_create
        self.block_name = block_name

    def forward(self, x):
        return x

    @staticmethod
    def create_from_str(s, no_create=False):
        assert PlainNetBasicBlockClass.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('PlainNetBasicBlockClass('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        return PlainNetBasicBlockClass(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('PlainNetBasicBlockClass(') and s[-1] == ')':
            return True
        else:
            return False


class AdaptiveAvgPool(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels,
                 output_size,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(AdaptiveAvgPool, self).__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels * output_size**2
        self.output_size = output_size
        self.block_name = block_name
        if not no_create:
            self.netblock = nn.AdaptiveAvgPool2d(
                output_size=(self.output_size, self.output_size))

    def forward(self, x):
        return self.netblock(x)

    @staticmethod
    def create_from_str(s, no_create=False):
        assert AdaptiveAvgPool.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('AdaptiveAvgPool('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        output_size = int(param_str_split[1])
        return AdaptiveAvgPool(
            out_channels=out_channels,
            output_size=output_size,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('AdaptiveAvgPool(') and s[-1] == ')':
            return True
        else:
            return False


class BN(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels=None,
                 copy_from=None,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(BN, self).__init__(**kwargs)
        self.block_name = block_name
        if copy_from is not None:
            assert isinstance(copy_from, nn.BatchNorm2d)
            self.in_channels = copy_from.weight.shape[0]
            self.out_channels = copy_from.weight.shape[0]
            assert out_channels is None or out_channels == self.out_channels
            self.netblock = copy_from

        else:
            self.in_channels = out_channels
            self.out_channels = out_channels
            if no_create:
                return
            else:
                self.netblock = nn.BatchNorm2d(num_features=self.out_channels)

    def forward(self, x):
        return self.netblock(x)

    @staticmethod
    def create_from_str(s, no_create=False):
        assert BN.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('BN('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        out_channels = int(param_str)
        return BN(
            out_channels=out_channels,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('BN(') and s[-1] == ')':
            return True
        else:
            return False


class ConvDW(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(ConvDW, self).__init__(**kwargs)
        self.block_name = block_name

        self.use_weight_mean_zero_constrain = False

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            assert self.in_channels == self.out_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride

            self.netblock = copy_from
        else:

            self.in_channels = out_channels
            self.out_channels = out_channels
            self.stride = stride
            self.kernel_size = kernel_size

            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                    or self.stride == 0:
                return
            else:
                self.netblock = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=False,
                    groups=self.in_channels)

    def forward(self, x):
        output = self.netblock(x)
        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert ConvDW.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('ConvDW('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        out_channels = int(split_str[0])
        kernel_size = int(split_str[1])
        stride = int(split_str[2])
        return ConvDW(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('ConvDW(') and s[-1] == ')':
            return True
        else:
            return False


class ConvKX(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 kernel_size=None,
                 stride=None,
                 copy_from=None,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(ConvKX, self).__init__(**kwargs)
        self.block_name = block_name
        self.use_weight_mean_zero_constrain = False

        if copy_from is not None:
            assert isinstance(copy_from, nn.Conv2d)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.kernel_size = copy_from.kernel_size[0]
            self.stride = copy_from.stride[0]
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels
            assert kernel_size is None or kernel_size == self.kernel_size
            assert stride is None or stride == self.stride

            self.netblock = copy_from
        else:

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            self.kernel_size = kernel_size

            self.padding = (self.kernel_size - 1) // 2
            if no_create or self.in_channels == 0 or self.out_channels == 0 or self.kernel_size == 0 \
                    or self.stride == 0:
                return
            else:
                self.netblock = nn.Conv2d(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    stride=self.stride,
                    padding=self.padding,
                    bias=False)

    def forward(self, x):
        output = self.netblock(x)

        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert ConvKX.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('ConvKX('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        kernel_size = int(split_str[2])
        stride = int(split_str[3])
        return ConvKX(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('ConvKX(') and s[-1] == ')':
            return True
        else:
            return False


class Flatten(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(Flatten, self).__init__(**kwargs)
        self.block_name = block_name
        self.in_channels = out_channels
        self.out_channels = out_channels

    def forward(self, x):
        return torch.flatten(x, 1)

    @staticmethod
    def create_from_str(s, no_create=False):
        assert Flatten.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('Flatten('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return Flatten(
            out_channels=out_channels,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('Flatten(') and s[-1] == ')':
            return True
        else:
            return False


class Linear(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=None,
                 out_channels=None,
                 bias=None,
                 copy_from=None,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.block_name = block_name

        if copy_from is not None:
            assert isinstance(copy_from, nn.Linear)
            self.in_channels = copy_from.in_channels
            self.out_channels = copy_from.out_channels
            self.bias = copy_from.bias
            assert in_channels is None or in_channels == self.in_channels
            assert out_channels is None or out_channels == self.out_channels
            assert bias is None or bias == self.bias

            self.netblock = copy_from
        else:

            self.in_channels = in_channels
            self.out_channels = out_channels
            self.bias = bias
            if not no_create:
                self.netblock = nn.Linear(
                    self.in_channels, self.out_channels, bias=self.bias)

    def forward(self, x):
        return self.netblock(x)

    @staticmethod
    def create_from_str(s, no_create=False):
        assert Linear.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('Linear('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        split_str = param_str.split(',')
        in_channels = int(split_str[0])
        out_channels = int(split_str[1])
        bias = int(split_str[2])

        return Linear(
            in_channels=in_channels,
            out_channels=out_channels,
            bias=bias == 1,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('Linear(') and s[-1] == ')':
            return True
        else:
            return False


class MaxPool(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels,
                 kernel_size,
                 stride,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(MaxPool, self).__init__(**kwargs)
        self.block_name = block_name
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        if not no_create:
            self.netblock = nn.MaxPool2d(
                kernel_size=self.kernel_size,
                stride=self.stride,
                padding=self.padding)

    def forward(self, x):
        return self.netblock(x)

    @staticmethod
    def create_from_str(s, no_create=False):
        assert MaxPool.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('MaxPool('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        kernel_size = int(param_str_split[1])
        stride = int(param_str_split[2])
        return MaxPool(
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('MaxPool(') and s[-1] == ')':
            return True
        else:
            return False


class MultiSumBlock(PlainNetBasicBlockClass):

    def __init__(self,
                 inner_block_list,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(MultiSumBlock, self).__init__(**kwargs)
        self.block_name = block_name
        self.inner_block_list = inner_block_list
        if not no_create:
            self.inner_module_list = nn.ModuleList(inner_block_list)
        self.in_channels = np.max([x.in_channels for x in inner_block_list])
        self.out_channels = np.max([x.out_channels for x in inner_block_list])

        res = 1024
        res = self.inner_block_list[0].get_output_resolution(res)
        self.stride = 1024 // res

    def forward(self, x):
        output = self.inner_block_list[0](x)

        for inner_block in self.inner_block_list[1:]:
            output2 = inner_block(x)
            output = output + output2

        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert MultiSumBlock.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('MultiSumBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_s = param_str

        the_inner_block_list = []
        while len(the_s) > 0:
            tmp_block_list, remaining_s = _create_netblock_list_from_str_(
                the_s, no_create=no_create)
            the_s = remaining_s
            if tmp_block_list is None:
                pass
            elif len(tmp_block_list) == 1:
                the_inner_block_list.append(tmp_block_list[0])
            else:
                the_inner_block_list.append(
                    Sequential(
                        inner_block_list=tmp_block_list, no_create=no_create))
        pass  # end while

        if len(the_inner_block_list) == 0:
            return None, s[idx + 1:]

        return MultiSumBlock(
            inner_block_list=the_inner_block_list,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('MultiSumBlock(') and s[-1] == ')':
            return True
        else:
            return False


class RELU(PlainNetBasicBlockClass):

    def __init__(self,
                 out_channels,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(RELU, self).__init__(**kwargs)
        self.block_name = block_name
        self.in_channels = out_channels
        self.out_channels = out_channels

    def forward(self, x):
        return F.relu(x)

    @staticmethod
    def create_from_str(s, no_create=False):
        assert RELU.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('RELU('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        out_channels = int(param_str)
        return RELU(
            out_channels=out_channels,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('RELU(') and s[-1] == ')':
            return True
        else:
            return False


class ResBlock(PlainNetBasicBlockClass):
    '''
    ResBlock(in_channles, inner_blocks_str). If in_channels is missing, use inner_block_list[0].in_channels as in_channels
    '''

    def __init__(self,
                 inner_block_list,
                 in_channels=None,
                 stride=None,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(ResBlock, self).__init__(**kwargs)
        self.block_name = block_name
        self.inner_block_list = inner_block_list
        self.stride = stride
        if not no_create:
            self.inner_module_list = nn.ModuleList(inner_block_list)

        if in_channels is None:
            self.in_channels = inner_block_list[0].in_channels
        else:
            self.in_channels = in_channels
        self.out_channels = max(self.in_channels,
                                inner_block_list[-1].out_channels)

        if self.stride is None:
            tmp_input_res = 1024
            tmp_output_res = self.get_output_resolution(tmp_input_res)
            self.stride = tmp_input_res // tmp_output_res

    def forward(self, x):
        if self.stride > 1:
            downsampled_x = F.avg_pool2d(
                x,
                kernel_size=self.stride + 1,
                stride=self.stride,
                padding=self.stride // 2)
        else:
            downsampled_x = x

        if len(self.inner_block_list) == 0:
            return downsampled_x

        output = x
        for inner_block in self.inner_block_list:
            output = inner_block(output)
        output = output + downsampled_x

        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert ResBlock.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        the_stride = None
        param_str = s[len('ResBlock('):idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        first_comma_index = param_str.find(',')
        if first_comma_index < 0 or not param_str[0:first_comma_index].isdigit(
        ):  # cannot parse in_channels, missing, use default
            in_channels = None
            the_inner_block_list, remaining_s = _create_netblock_list_from_str_(
                param_str, no_create=no_create)
        else:
            in_channels = int(param_str[0:first_comma_index])
            param_str = param_str[first_comma_index + 1:]
            second_comma_index = param_str.find(',')
            if second_comma_index < 0 or not param_str[
                    0:second_comma_index].isdigit():
                the_inner_block_list, remaining_s = _create_netblock_list_from_str_(
                    param_str, no_create=no_create)
            else:
                the_stride = int(param_str[0:second_comma_index])
                param_str = param_str[second_comma_index + 1:]
                the_inner_block_list, remaining_s = _create_netblock_list_from_str_(
                    param_str, no_create=no_create)
            pass
        pass

        assert len(remaining_s) == 0
        if the_inner_block_list is None or len(the_inner_block_list) == 0:
            return None, s[idx + 1:]
        return ResBlock(
            inner_block_list=the_inner_block_list,
            in_channels=in_channels,
            stride=the_stride,
            no_create=no_create,
            block_name=tmp_block_name), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('ResBlock(') and s[-1] == ')':
            return True
        else:
            return False


class Sequential(PlainNetBasicBlockClass):

    def __init__(self,
                 inner_block_list,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(Sequential, self).__init__(**kwargs)
        self.block_name = block_name
        self.inner_block_list = inner_block_list
        if not no_create:
            self.inner_module_list = nn.ModuleList(inner_block_list)
        self.in_channels = inner_block_list[0].in_channels
        self.out_channels = inner_block_list[-1].out_channels

        res = 1024
        for block in self.inner_block_list:
            res = block.get_output_resolution(res)

        self.stride = 1024 // res

    def forward(self, x):

        output = x
        for inner_block in self.inner_block_list:
            output = inner_block(output)

        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert Sequential.is_instance_from_str(s)
        the_right_paraen_idx = _get_right_parentheses_index_(s)
        param_str = s[len('Sequential(') + 1:the_right_paraen_idx]
        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        the_inner_block_list, remaining_s = _create_netblock_list_from_str_(
            param_str, no_create=no_create)
        assert len(remaining_s) == 0
        if the_inner_block_list is None or len(the_inner_block_list) == 0:
            return None, ''
        return Sequential(
            inner_block_list=the_inner_block_list,
            no_create=no_create,
            block_name=tmp_block_name), ''

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('Sequential('):
            return True
        else:
            return False


'''
Super Blocks
'''


class SuperResKXKX(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 kernel_size=3,
                 stride=1,
                 expansion=1.0,
                 sublayers=1,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(SuperResKXKX, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.stride = stride
        self.sublayers = sublayers
        self.no_create = no_create
        self.block_name = block_name

        self.shortcut_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()

        for layerID in range(self.sublayers):
            if layerID == 0:
                current_in_channels = self.in_channels
                current_out_channels = self.out_channels
                current_stride = self.stride
                current_kernel_size = self.kernel_size
            else:
                current_in_channels = self.out_channels
                current_out_channels = self.out_channels
                current_stride = 1
                current_kernel_size = self.kernel_size

            current_expansion_channel = int(
                round(current_out_channels * self.expansion))

            the_conv_block = nn.Sequential(
                nn.Conv2d(
                    current_in_channels,
                    current_expansion_channel,
                    kernel_size=current_kernel_size,
                    stride=current_stride,
                    padding=(current_kernel_size - 1) // 2,
                    bias=False),
                nn.BatchNorm2d(current_expansion_channel),
                nn.ReLU(),
                nn.Conv2d(
                    current_expansion_channel,
                    current_out_channels,
                    kernel_size=current_kernel_size,
                    stride=1,
                    padding=(current_kernel_size - 1) // 2,
                    bias=False),
                nn.BatchNorm2d(current_out_channels),
            )
            self.conv_list.append(the_conv_block)

            if current_stride == 1 and current_in_channels == current_out_channels:
                shortcut = nn.Sequential()
            else:
                shortcut = nn.Sequential(
                    nn.Conv2d(
                        current_in_channels,
                        current_out_channels,
                        kernel_size=1,
                        stride=current_stride,
                        padding=0,
                        bias=False), nn.BatchNorm2d(current_out_channels))
            self.shortcut_list.append(shortcut)
        pass  # end for

    def forward(self, x):
        output = x
        for block, shortcut in zip(self.conv_list, self.shortcut_list):
            conv_output = block(output)
            output = conv_output + shortcut(output)
            output = F.relu(output)
        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert SuperResKXKX.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('SuperResKXKX('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        kernel_size = int(param_str_split[2])
        stride = int(param_str_split[3])
        expansion = float(param_str_split[4])
        sublayers = int(param_str_split[5])

        return SuperResKXKX(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expansion=expansion,
            sublayers=sublayers,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('SuperResKXKX(') and s[-1] == ')':
            return True
        else:
            return False


class SuperResK1KX(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 kernel_size=3,
                 stride=1,
                 expansion=1.0,
                 sublayers=1,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(SuperResK1KX, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.stride = stride
        self.sublayers = sublayers
        self.no_create = no_create
        self.block_name = block_name

        self.shortcut_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()

        for layerID in range(self.sublayers):
            if layerID == 0:
                current_in_channels = self.in_channels
                current_out_channels = self.out_channels
                current_stride = self.stride
                current_kernel_size = self.kernel_size
            else:
                current_in_channels = self.out_channels
                current_out_channels = self.out_channels
                current_stride = 1
                current_kernel_size = self.kernel_size

            current_expansion_channel = int(
                round(current_out_channels * self.expansion))

            the_conv_block = nn.Sequential(
                nn.Conv2d(
                    current_in_channels,
                    current_expansion_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(current_expansion_channel),
                nn.ReLU(),
                nn.Conv2d(
                    current_expansion_channel,
                    current_out_channels,
                    kernel_size=current_kernel_size,
                    stride=current_stride,
                    padding=(current_kernel_size - 1) // 2,
                    bias=False),
                nn.BatchNorm2d(current_out_channels),
            )
            self.conv_list.append(the_conv_block)

            if current_stride == 1 and current_in_channels == current_out_channels:
                shortcut = nn.Sequential()
            else:
                shortcut = nn.Sequential(
                    nn.Conv2d(
                        current_in_channels,
                        current_out_channels,
                        kernel_size=1,
                        stride=current_stride,
                        padding=0,
                        bias=False), nn.BatchNorm2d(current_out_channels))
            self.shortcut_list.append(shortcut)
        pass  # end for

    def forward(self, x):
        output = x
        for block, shortcut in zip(self.conv_list, self.shortcut_list):
            conv_output = block(output)
            output = conv_output + shortcut(output)
            output = F.relu(output)
        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert SuperResK1KX.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('SuperResK1KX('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        kernel_size = int(param_str_split[2])
        stride = int(param_str_split[3])
        expansion = float(param_str_split[4])
        sublayers = int(param_str_split[5])

        return SuperResK1KX(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expansion=expansion,
            sublayers=sublayers,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('SuperResK1KX(') and s[-1] == ')':
            return True
        else:
            return False


class SuperResK1KXK1(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 kernel_size=3,
                 stride=1,
                 expansion=1.0,
                 sublayers=1,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(SuperResK1KXK1, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.stride = stride
        self.sublayers = sublayers
        self.no_create = no_create
        self.block_name = block_name

        self.shortcut_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()

        for layerID in range(self.sublayers):
            if layerID == 0:
                current_in_channels = self.in_channels
                current_out_channels = self.out_channels
                current_stride = self.stride
                current_kernel_size = self.kernel_size
            else:
                current_in_channels = self.out_channels
                current_out_channels = self.out_channels
                current_stride = 1
                current_kernel_size = self.kernel_size

            current_expansion_channel = int(
                round(current_out_channels * self.expansion))

            the_conv_block = nn.Sequential(
                nn.Conv2d(
                    current_in_channels,
                    current_expansion_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(current_expansion_channel),
                nn.ReLU(),
                nn.Conv2d(
                    current_expansion_channel,
                    current_expansion_channel,
                    kernel_size=current_kernel_size,
                    stride=current_stride,
                    padding=(current_kernel_size - 1) // 2,
                    bias=False),
                nn.BatchNorm2d(current_expansion_channel),
                nn.ReLU(),
                nn.Conv2d(
                    current_expansion_channel,
                    current_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(current_out_channels),
            )
            self.conv_list.append(the_conv_block)

            if current_stride == 1 and current_in_channels == current_out_channels:
                shortcut = nn.Sequential()
            else:
                shortcut = nn.Sequential(
                    nn.Conv2d(
                        current_in_channels,
                        current_out_channels,
                        kernel_size=1,
                        stride=current_stride,
                        padding=0,
                        bias=False), nn.BatchNorm2d(current_out_channels))
            self.shortcut_list.append(shortcut)
        pass  # end for

    def forward(self, x):
        output = x
        for block, shortcut in zip(self.conv_list, self.shortcut_list):
            conv_output = block(output)
            output = conv_output + shortcut(output)
            output = F.relu(output)
        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert SuperResK1KXK1.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('SuperResK1KXK1('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        kernel_size = int(param_str_split[2])
        stride = int(param_str_split[3])
        expansion = float(param_str_split[4])
        sublayers = int(param_str_split[5])

        return SuperResK1KXK1(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expansion=expansion,
            sublayers=sublayers,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('SuperResK1KXK1(') and s[-1] == ')':
            return True
        else:
            return False


class SuperResK1DWK1(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 kernel_size=3,
                 stride=1,
                 expansion=1.0,
                 sublayers=1,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(SuperResK1DWK1, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        self.stride = stride
        self.sublayers = sublayers
        self.no_create = no_create
        self.block_name = block_name

        self.shortcut_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()

        for layerID in range(self.sublayers):
            if layerID == 0:
                current_in_channels = self.in_channels
                current_out_channels = self.out_channels
                current_stride = self.stride
                current_kernel_size = self.kernel_size
            else:
                current_in_channels = self.out_channels
                current_out_channels = self.out_channels
                current_stride = 1
                current_kernel_size = self.kernel_size

            current_expansion_channel = int(
                round(current_out_channels * self.expansion))

            the_conv_block = nn.Sequential(
                nn.Conv2d(
                    current_in_channels,
                    current_expansion_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(current_expansion_channel),
                nn.ReLU(),
                nn.Conv2d(
                    current_expansion_channel,
                    current_expansion_channel,
                    kernel_size=current_kernel_size,
                    stride=current_stride,
                    padding=(current_kernel_size - 1) // 2,
                    bias=False,
                    groups=current_expansion_channel),
                nn.BatchNorm2d(current_expansion_channel),
                nn.ReLU(),
                nn.Conv2d(
                    current_expansion_channel,
                    current_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(current_out_channels),
            )
            self.conv_list.append(the_conv_block)

            if current_stride == 1 and current_in_channels == current_out_channels:
                shortcut = nn.Sequential()
            else:
                shortcut = nn.Sequential(
                    nn.Conv2d(
                        current_in_channels,
                        current_out_channels,
                        kernel_size=1,
                        stride=current_stride,
                        padding=0,
                        bias=False), nn.BatchNorm2d(current_out_channels))
            self.shortcut_list.append(shortcut)
        pass  # end for

    def forward(self, x):
        output = x
        for block, shortcut in zip(self.conv_list, self.shortcut_list):
            conv_output = block(output)
            output = conv_output + shortcut(output)
            output = F.relu(output)
        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert SuperResK1DWK1.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('SuperResK1DWK1('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        kernel_size = int(param_str_split[2])
        stride = int(param_str_split[3])
        expansion = float(param_str_split[4])
        sublayers = int(param_str_split[5])

        return SuperResK1DWK1(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expansion=expansion,
            sublayers=sublayers,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('SuperResK1DWK1(') and s[-1] == ')':
            return True
        else:
            return False


class SuperResK1DW(PlainNetBasicBlockClass):

    def __init__(self,
                 in_channels=0,
                 out_channels=0,
                 kernel_size=3,
                 stride=1,
                 expansion=1.0,
                 sublayers=1,
                 no_create=False,
                 block_name=None,
                 **kwargs):
        super(SuperResK1DW, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.expansion = expansion
        assert abs(expansion - 1) < 1e-6
        self.stride = stride
        self.sublayers = sublayers
        self.no_create = no_create
        self.block_name = block_name

        self.shortcut_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()

        for layerID in range(self.sublayers):
            if layerID == 0:
                current_in_channels = self.in_channels
                current_out_channels = self.out_channels
                current_stride = self.stride
                current_kernel_size = self.kernel_size
            else:
                current_in_channels = self.out_channels
                current_out_channels = self.out_channels
                current_stride = 1
                current_kernel_size = self.kernel_size

            current_expansion_channel = int(
                round(current_out_channels * self.expansion))

            the_conv_block = nn.Sequential(
                nn.Conv2d(
                    current_in_channels,
                    current_out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False),
                nn.BatchNorm2d(current_expansion_channel),
                nn.ReLU(),
                nn.Conv2d(
                    current_out_channels,
                    current_out_channels,
                    kernel_size=current_kernel_size,
                    stride=current_stride,
                    padding=(current_kernel_size - 1) // 2,
                    bias=False,
                    groups=current_out_channels),
                nn.BatchNorm2d(current_out_channels),
            )
            self.conv_list.append(the_conv_block)

            if current_stride == 1 and current_in_channels == current_out_channels:
                shortcut = nn.Sequential()
            else:
                shortcut = nn.Sequential(
                    nn.Conv2d(
                        current_in_channels,
                        current_out_channels,
                        kernel_size=1,
                        stride=current_stride,
                        padding=0,
                        bias=False), nn.BatchNorm2d(current_out_channels))
            self.shortcut_list.append(shortcut)
        pass  # end for

    def forward(self, x):
        output = x
        for block, shortcut in zip(self.conv_list, self.shortcut_list):
            conv_output = block(output)
            output = conv_output + shortcut(output)
            output = F.relu(output)
        return output

    @staticmethod
    def create_from_str(s, no_create=False):
        assert SuperResK1DW.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len('SuperResK1DW('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        kernel_size = int(param_str_split[2])
        stride = int(param_str_split[3])
        expansion = float(param_str_split[4])
        sublayers = int(param_str_split[5])

        return SuperResK1DW(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            expansion=expansion,
            sublayers=sublayers,
            block_name=tmp_block_name,
            no_create=no_create), s[idx + 1:]

    @staticmethod
    def is_instance_from_str(s):
        if s.startswith('SuperResK1DW(') and s[-1] == ')':
            return True
        else:
            return False


_all_netblocks_dict_ = {
    'AdaptiveAvgPool': AdaptiveAvgPool,
    'BN': BN,
    'ConvDW': ConvDW,
    'ConvKX': ConvKX,
    'Flatten': Flatten,
    'Linear': Linear,
    'MaxPool': MaxPool,
    'MultiSumBlock': MultiSumBlock,
    'PlainNetBasicBlockClass': PlainNetBasicBlockClass,
    'RELU': RELU,
    'ResBlock': ResBlock,
    'Sequential': Sequential,
    'SuperResKXKX': SuperResKXKX,
    'SuperResK1KXK1': SuperResK1KXK1,
    'SuperResK1DWK1': SuperResK1DWK1,
    'SuperResK1KX': SuperResK1KX,
    'SuperResK1DW': SuperResK1DW,
}


@BACKBONES.register_module
class PlainNet(nn.Module):

    def __init__(self,
                 plainnet_struct_idx=None,
                 num_classes=0,
                 no_create=False,
                 **kwargs):
        super(PlainNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.plainnet_struct = plainnet_struct_dict[plainnet_struct_idx]
        the_s = self.plainnet_struct  # type: str

        block_list, remaining_s = _create_netblock_list_from_str_(
            the_s, no_create=no_create)
        assert len(remaining_s) == 0
        if isinstance(block_list[-1], AdaptiveAvgPool):
            self.adptive_avg_pool = block_list[-1]
            block_list.pop(-1)
        else:
            self.adptive_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)  # register

        self.last_channels = self.adptive_avg_pool.out_channels

        if num_classes > 0:
            self.fc_linear = nn.Linear(
                self.last_channels, self.num_classes, bias=True)
        else:
            self.fc_linear = None

        self.plainnet_struct = str(self) + str(self.adptive_avg_pool)
        self.zero_init_residual = False

        self.default_pretrained_model_path = model_urls[self.__class__.__name__
                                                        + plainnet_struct_idx]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)

        if self.fc_linear is not None:
            bs = output.size(0)
            output = self.adptive_avg_pool(output)
            output = output.view(bs, -1)
            output = self.fc_linear(output)

        return [output]
