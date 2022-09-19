# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from easycv.framework.errors import NotImplementedError
from easycv.models.builder import NECKS


@NECKS.register_module()
class SFP(nn.Module):
    r"""Simple Feature Pyramid.
    This is an implementation of paper `Exploring Plain Vision Transformer Backbones for Object Detection <https://arxiv.org/abs/2203.16527>`_.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
            conv. Default: False.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = SFP(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factors,
                 num_outs,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(SFP, self).__init__()
        dim = in_channels
        self.out_channels = out_channels
        self.scale_factors = scale_factors
        self.num_ins = len(scale_factors)
        self.num_outs = num_outs

        self.stages = []
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, 2, stride=2, padding=0),
                    nn.GroupNorm(1, dim // 2, eps=1e-6),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        dim // 2, dim // 4, 2, stride=2, padding=0)
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, 2, stride=2, padding=0)
                ]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
            else:
                raise NotImplementedError(
                    f'scale_factor={scale} is not supported yet.')

            layers.extend([
                ConvModule(
                    out_dim,
                    out_channels,
                    1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False),
                ConvModule(
                    out_channels,
                    out_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
            ])

            layers = nn.Sequential(*layers)
            self.add_module(f'sfp_{idx}', layers)
            self.stages.append(layers)

    def init_weights(self):
        pass

    def forward(self, inputs):
        """Forward function."""
        features = inputs[0]
        outs = []

        # part 1: build simple feature pyramid
        for stage in self.stages:
            outs.append(stage(features))

        # part 2: add extra levels
        if self.num_outs > self.num_ins:
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)
