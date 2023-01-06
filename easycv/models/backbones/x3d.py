# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math

import torch.nn as nn
from fvcore.nn.weight_init import c2_msra_fill

from easycv.models.utils.video_model_stem import VideoModelStem
from easycv.models.utils.x3d_transformer import ResStage
from ..registry import BACKBONES

_MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3)}
NUM_GROUPS = 1
WIDTH_PER_GROUP = 64
INPUT_CHANNEL_NUM = [3]
CHANNELWISE_3x3x3 = True
NONLOCAL_LOCATION = [[[]], [[]], [[]], [[]]]
NONLOCAL_GROUP = [[1], [1], [1], [1]]
NONLOCAL_POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]
NONLOCAL_INSTANTIATION = 'dot_product'
RESNET_SPATIAL_DILATIONS = [[1], [1], [1], [1]]


class X3DHead(nn.Module):
    """
    X3D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_inner,
        dim_out,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func='softmax',
        inplace_relu=True,
        eps=1e-5,
        bn_mmt=0.1,
        norm_module=nn.BatchNorm3d,
        bn_lin5_on=False,
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        X3DHead takes a 5-dim feature tensor (BxCxTxHxW) as input.

        Args:
            dim_in (float): the channel dimension C of the input.
            num_classes (int): the channel dimensions of the output.
            pool_size (float): a single entry list of kernel size for
                spatiotemporal pooling for the TxHxW dimensions.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                default is nn.BatchNorm3d.
            bn_lin5_on (bool): if True, perform normalization on the features
                before the classifier.
        """
        super(X3DHead, self).__init__()
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes
        self.act_func = act_func
        self.eps = eps
        self.bn_mmt = bn_mmt
        self.inplace_relu = inplace_relu
        self.bn_lin5_on = bn_lin5_on
        self._construct_head(dim_in, dim_inner, dim_out, norm_module)

    def _construct_head(self, dim_in, dim_inner, dim_out, norm_module):

        self.conv_5 = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.conv_5_bn = norm_module(
            num_features=dim_inner, eps=self.eps, momentum=self.bn_mmt)
        self.conv_5_relu = nn.ReLU(self.inplace_relu)

        if self.pool_size is None:
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = nn.AvgPool3d(self.pool_size, stride=1)

        self.lin_5 = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=(1, 1, 1),
            stride=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        if self.bn_lin5_on:
            self.lin_5_bn = norm_module(
                num_features=dim_out, eps=self.eps, momentum=self.bn_mmt)
        self.lin_5_relu = nn.ReLU(self.inplace_relu)

        if self.dropout_rate > 0.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_out, self.num_classes, bias=True)

        # Softmax for evaluation and testing.
        if self.act_func == 'softmax':
            self.act = nn.Softmax(dim=4)
        elif self.act_func == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError('{} is not supported as an activation'
                                      'function.'.format(self.act_func))

    def forward(self, inputs):
        # In its current design the X3D head is only useable for a single
        # pathway input.
        # assert len(inputs) == 1, "Input tensor does not contain 1 pathway"
        x = self.conv_5(inputs)
        x = self.conv_5_bn(x)
        x = self.conv_5_relu(x)

        x = self.avg_pool(x)

        x = self.lin_5(x)
        if self.bn_lin5_on:
            x = self.lin_5_bn(x)
        x = self.lin_5_relu(x)

        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        # Perform dropout.
        if hasattr(self, 'dropout'):
            x = self.dropout(x)
        x = self.projection(x)

        # Performs fully convlutional inference.
        if not self.training:
            x = self.act(x)
            x = x.mean([1, 2, 3])

        x = x.view(x.shape[0], -1)
        return x


@BACKBONES.register_module
class X3D(nn.Module):

    def __init__(
            self,
            width_factor=2.0,
            depth_factor=2.2,
            bottlneck_factor=2.25,
            dim_c5=2048,
            dim_c1=12,
            #  train_crop_size=160,
            num_classes=400,
            num_frames=4,
            pretrained=None):
        super(X3D, self).__init__()

        self.width_factor = width_factor
        self.depth_factor = depth_factor
        self.bottlneck_factor = bottlneck_factor
        self.dim_c5 = dim_c5
        self.dim_c1 = dim_c1
        # self.train_crop_size = train_crop_size
        self.num_classes = num_classes
        self.num_frames = num_frames
        self.pretrained = pretrained

        self.norm_module = nn.BatchNorm3d
        self.num_pathways = 1

        exp_stage = 2.0

        SCALE_RES2 = False
        self.dim_res2 = (
            self._round_width(self.dim_c1, exp_stage, divisor=8)
            if SCALE_RES2 else self.dim_c1)
        self.dim_res3 = self._round_width(self.dim_res2, exp_stage, divisor=8)
        self.dim_res4 = self._round_width(self.dim_res3, exp_stage, divisor=8)
        self.dim_res5 = self._round_width(self.dim_res4, exp_stage, divisor=8)

        self.block_basis = [
            # blocks, c, stride
            [1, self.dim_res2, 2],
            [2, self.dim_res3, 2],
            [5, self.dim_res4, 2],
            [3, self.dim_res5, 2],
        ]
        self._construct_network()
        # init_weights(self)

    def init_weights(self, fc_init_std=0.01, zero_init_final_bn=True):
        """
        Performs ResNet style weight initialization.
        Args:
            fc_init_std (float): the expected standard deviation for fc layer.
            zero_init_final_bn (bool): if True, zero initialize the final bn for
                every bottleneck.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                """
                Follow the initialization method proposed in:
                {He, Kaiming, et al.
                "Delving deep into rectifiers: Surpassing human-level
                performance on imagenet classification."
                arXiv preprint arXiv:1502.01852 (2015)}
                """
                c2_msra_fill(m)
            elif isinstance(m, nn.BatchNorm3d):
                if (hasattr(m, 'transform_final_bn') and m.transform_final_bn
                        and zero_init_final_bn):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                if m.weight is not None:
                    m.weight.data.fill_(batchnorm_weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=fc_init_std)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _round_width(self, width, multiplier, min_depth=8, divisor=8):
        """Round width of filters based on width multiplier."""
        if not multiplier:
            return width

        width *= multiplier
        min_depth = min_depth or divisor
        new_filters = max(min_depth,
                          int(width + divisor / 2) // divisor * divisor)
        if new_filters < 0.9 * width:
            new_filters += divisor
        return int(new_filters)

    def _round_repeats(self, repeats, multiplier):
        """Round number of layers based on depth multiplier."""
        multiplier = multiplier
        if not multiplier:
            return repeats
        return int(math.ceil(multiplier * repeats))

    def _construct_network(self):

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[50]
        num_groups = NUM_GROUPS
        width_per_group = WIDTH_PER_GROUP
        dim_inner = num_groups * width_per_group

        w_mul = self.width_factor
        d_mul = self.depth_factor
        dim_res1 = self._round_width(self.dim_c1, w_mul)

        temp_kernel = [
            [[5]],  # conv1 temporal kernels.
            [[3]],  # res2 temporal kernels.
            [[3]],  # res3 temporal kernels.
            [[3]],  # res4 temporal kernels.
            [[3]],  # res5 temporal kernels.
        ]
        self.s1 = VideoModelStem(
            dim_in=INPUT_CHANNEL_NUM,
            dim_out=[dim_res1],
            kernel=[temp_kernel[0][0] + [3, 3]],
            stride=[[1, 2, 2]],
            padding=[[temp_kernel[0][0][0] // 2, 1, 1]],
            norm_module=self.norm_module,
            stem_func_name='x3d_stem',
        )

        # blob_in = s1
        dim_in = dim_res1
        for stage, block in enumerate(self.block_basis):
            dim_out = self._round_width(block[1], w_mul)
            dim_inner = int(self.bottlneck_factor * dim_out)

            n_rep = self._round_repeats(block[0], d_mul)
            prefix = 's{}'.format(stage +
                                  2)  # start w res2 to follow convention
            s = ResStage(
                dim_in=[dim_in],
                dim_out=[dim_out],
                dim_inner=[dim_inner],
                temp_kernel_sizes=temp_kernel[1],
                stride=[block[2]],
                num_blocks=[n_rep],
                num_groups=[dim_inner] if CHANNELWISE_3x3x3 else [num_groups],
                num_block_temp_kernel=[n_rep],
                nonlocal_inds=NONLOCAL_LOCATION[0],
                nonlocal_group=NONLOCAL_GROUP[0],
                nonlocal_pool=NONLOCAL_POOL[0],
                instantiation=NONLOCAL_INSTANTIATION,
                trans_func_name='x3d_transform',
                stride_1x1=False,
                norm_module=self.norm_module,
                dilation=RESNET_SPATIAL_DILATIONS[stage],
                drop_connect_rate=0.0 * (stage + 2) /
                (len(self.block_basis) + 1),
            )
            dim_in = dim_out
            self.add_module(prefix, s)

    def forward(self, x):
        for step, module in enumerate(self.children()):
            x = module(x)
        return x
