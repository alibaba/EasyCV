# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm

from ..modelzoo import resnet as model_urls
from ..registry import BACKBONES
from ..utils import FReLU, build_conv_layer, build_norm_layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 frelu=False):
        super(BasicBlock, self).__init__()

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.frelu = frelu
        if frelu:
            self.frelu_a = FReLU(planes)
            self.frelu_b = FReLU(planes)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        assert not with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        if self.frelu:
            out = self.frelu_a(out)
        else:
            out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        if self.frelu:
            out = self.frelu_b(out)
        else:
            out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 frelu=False):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg,
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.frelu = frelu
        if self.frelu:
            self.relu_a = FReLU(planes)
            self.relu_b = FReLU(planes)
            self.relu_c = FReLU(planes * self.expansion)
        else:
            self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            if self.frelu:
                out = self.relu_a(out)
            else:
                out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)
            if self.frelu:
                out = self.relu_b(out)
            else:
                out = self.relu(out)
            out = self.conv3(out)
            out = self.norm3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        if self.frelu:
            out = self.relu_c(out)
        else:
            out = self.relu(out)

        return out


def make_res_layer(
    block,
    inplanes,
    planes,
    blocks,
    stride=1,
    dilation=1,
    style='pytorch',
    avg_down=False,
    with_cp=False,
    conv_cfg=None,
    norm_cfg=dict(type='BN'),
    frelu=False,
    multi_grid=None,
    contract_dilation=False,
):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = []
        conv_stride = stride
        if avg_down:
            conv_stride = 1
            downsample.append(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False))
        downsample.extend([
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=conv_stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1]
        ])
        downsample = nn.Sequential(*downsample)

    if multi_grid is None:
        if dilation > 1 and contract_dilation:
            first_dilation = dilation // 2
        else:
            first_dilation = dilation
    else:
        first_dilation = multi_grid[0]
    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=first_dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            frelu=frelu))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation if multi_grid is None else multi_grid[i],
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                frelu=frelu))

    return nn.Sequential(*layers)


@BACKBONES.register_module
class ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        in_channels (int): Number of input image channels. Normally 3.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv.
            Default: False.
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck. Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        original_inplanes: start channel for first block, default=64
        stem_channels (int): Number of stem channels. Default: 64.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
        multi_grid (Sequence[int]|None): Multi grid dilation rates of last
            stage. Default: None.
        contract_dilation (bool): Whether contract first dilation of each layer
            Default: False.

    Example:
        >>> from easycv.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        10: (BasicBlock, (1, 1, 1, 1)),
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3, 4),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 num_classes=0,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=False,
                 with_cp=False,
                 frelu=False,
                 original_inplanes=64,
                 stem_channels=64,
                 zero_init_residual=False,
                 multi_grid=None,
                 contract_dilation=False):
        super(ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError('invalid depth {} for resnet'.format(depth))
        self.depth = depth
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages + 1
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.original_inplanes = original_inplanes
        self.stem_channels = stem_channels
        self.inplanes = stem_channels
        self.frelu = frelu
        self.multi_grid = multi_grid
        self.contract_dilation = contract_dilation

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            # multi grid is applied to last layer only
            stage_multi_grid = multi_grid if i == len(
                self.stage_blocks) - 1 else None
            planes = self.original_inplanes * 2**i
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                frelu=self.frelu,
                multi_grid=stage_multi_grid,
                contract_dilation=contract_dilation,
            )
            self.inplanes = planes * self.block.expansion
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

        self._freeze_stages()

        self.feat_dim = self.block.expansion * self.original_inplanes * 2**(
            len(self.stage_blocks) - 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if num_classes > 0:
            self.fc = nn.Linear(self.feat_dim, num_classes)

        self.default_pretrained_model_path = model_urls.get(
            self.__class__.__name__ + str(depth), None)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.frelu:
            relu = FReLU(stem_channels)
        else:
            relu = nn.ReLU(inplace=True)

        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1], relu,
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1], relu,
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1], relu)
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = relu

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    constant_init(m.norm3, 0)
                elif isinstance(m, BasicBlock):
                    constant_init(m.norm2, 0)

    def forward(self, x):
        outs = []
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)  # r50: 64x128x128
        if 0 in self.out_indices:
            outs.append(x)
        x = self.maxpool(x)  # r50: 64x56x56
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i + 1 in self.out_indices:
                outs.append(x)
        # r50: 1-256x56x56; 2-512x28x28; 3-1024x14x14; 4-2048x7x7

        if hasattr(self, 'fc'):
            bs = x.size(0)
            x = self.avgpool(x).view(bs, -1)
            x = self.fc(x)
            # outs.append(x)
            outs = [x]

        return outs

    def train(self, mode=True):
        super(ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


@BACKBONES.register_module()
class ResNetV1c(ResNet):
    """Compared to ResNet, ResNetV1c replaces the 7x7 conv in the input stem with three 3x3 convs.
    For more details please refer to <https://arxiv.org/abs/1812.01187>.
    """

    def __init__(self, **kwargs):
        super(ResNetV1c, self).__init__(
            deep_stem=True, avg_down=False, **kwargs)


@BACKBONES.register_module()
class ResNetV1d(ResNet):
    """Compared to ResNet, ResNetV1d replaces the 7x7 conv in the input stem with three 3x3 convs.
    And in the downsampling block, a 2x2 avg_pool with stride 2 is added before conv, whose stride is changed to 1.
    """

    def __init__(self, **kwargs):
        super(ResNetV1d, self).__init__(
            deep_stem=True, avg_down=True, **kwargs)
