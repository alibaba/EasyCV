# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import reduce

import torch
import torch.nn as nn
from packaging import version

from easycv.models.utils import GeMPooling, ResLayer
from ..backbones.hrnet import Bottleneck
from ..registry import NECKS
from ..utils import ConvModule, _init_weights, build_norm_layer


@NECKS.register_module
class LinearNeck(nn.Module):
    '''Linear neck: fc only
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 with_avg_pool=True,
                 with_norm=False):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)
        self.with_norm = with_norm

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2  # to fit vit model
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)

        x = self.fc(x.view(x.size(0), -1))
        if self.with_norm:
            x = nn.functional.normalize(x, p=2, dim=1)
        return [x]


@NECKS.register_module
class RetrivalNeck(nn.Module):
    '''RetrivalNeck: refer, Combination of Multiple Global Descriptors for Image Retrieval
         https://arxiv.org/pdf/1903.10663.pdf

       CGD feature : only use avg pool + gem pooling + max pooling, by pool -> fc -> norm -> concat -> norm
       Avg feature : use avg pooling, avg pool -> syncbn -> fc

       len(cgd_config) > 0: return  [CGD, Avg]
       len(cgd_config) = 0 : return [Avg]
    '''

    def __init__(
        self,
        in_channels,
        out_channels,
        with_avg_pool=True,
        cdg_config=[
            'G', 'M'
        ]):  # with_avg_pool=True, with_gem_pool=True,  with_norm=False):
        """ Init RetrivalNeck, faceid neck doesn't pool for input feature map, doesn't support dynamic input

        Args:
            in_channels: Int - input feature map channels
            out_channels: Int - output feature map channels
            with_avg_pool: bool do avg pool for BNneck or not
            cdg_config : list('G','M','S'), to configure output feature, CGD =  [gempooling] + [maxpooling] + [meanpooling],
                if len(cgd_config) > 0: return  [CGD, Avg]
                if len(cgd_config) = 0 : return [Avg]
        """
        super(RetrivalNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(in_channels, out_channels, bias=False)
        self.dropout = nn.Dropout(p=0.3)
        _, self.bn_output = build_norm_layer(dict(type='BN'), in_channels)
        # dict(type='SyncBN'), in_channels)

        self.cdg_config = cdg_config
        cgd_length = int(len(cdg_config))
        if cgd_length > 0:
            assert (out_channels % cgd_length == 0)
            if 'M' in cdg_config:
                self.mpool = nn.AdaptiveMaxPool2d((1, 1))
                self.fc_mx = nn.Linear(
                    in_channels, int(out_channels / cgd_length), bias=False)
            if 'S' in cdg_config:
                self.spool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc_sx = nn.Linear(
                    in_channels, int(out_channels / cgd_length), bias=False)
            if 'G' in cdg_config:
                self.gpool = GeMPooling()
                self.fc_gx = nn.Linear(
                    in_channels, int(out_channels / cgd_length), bias=False)

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2  # to fit vit model
        x = x[0]

        # BNNeck with avg pool
        if self.with_avg_pool:
            ax = self.avgpool(x)
        else:
            ax = x
        cls_x = self.bn_output(ax)
        cls_x = self.fc(cls_x.view(x.size(0), -1))
        cls_x = self.dropout(cls_x)

        if len(self.cdg_config) > 0:
            concat_list = []
            if 'S' in self.cdg_config:
                sx = self.spool(x).view(x.size(0), -1)
                sx = self.fc_sx(sx)
                sx = nn.functional.normalize(sx, p=2, dim=1)
                concat_list.append(sx)

            if 'G' in self.cdg_config:
                gx = self.gpool(x).view(x.size(0), -1)
                gx = self.fc_gx(gx)
                gx = nn.functional.normalize(gx, p=2, dim=1)
                concat_list.append(gx)

            if 'M' in self.cdg_config:
                mx = self.mpool(x).view(x.size(0), -1)
                mx = self.fc_mx(mx)
                mx = nn.functional.normalize(mx, p=2, dim=1)
                concat_list.append(mx)

            concatx = torch.cat(concat_list, dim=1)
            concatx = concatx.view(concatx.size(0), -1)
            # concatx = nn.functional.normalize(concatx, p=2, dim=1)
            return [concatx, cls_x]
        else:
            return [cls_x]


@NECKS.register_module
class FaceIDNeck(nn.Module):
    '''FaceID neck: Include BN, dropout, flatten, linear, bn
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 map_shape=1,
                 dropout_ratio=0.4,
                 with_norm=False,
                 bn_type='SyncBN'):
        """ Init FaceIDNeck, faceid neck doesn't pool for input feature map, doesn't support dynamic input

        Args:
            in_channels: Int - input feature map channels
            out_channels: Int - output feature map channels
            map_shape: Int or list(int,...), input feature map (w,h) or w when w=h,
            dropout_ratio : float, drop out ratio
            with_norm : normalize output feature or not
            bn_type : SyncBN or BN
        """
        super(FaceIDNeck, self).__init__()

        if version.parse(torch.__version__) < version.parse('1.4.0'):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        # self.bn_input = nn.BatchNorm2d(in_channels)
        _, self.bn_input = build_norm_layer(dict(type=bn_type), in_channels)
        self.dropout = nn.Dropout(p=dropout_ratio)

        if type(map_shape) == list:
            in_ = int(reduce(lambda x, y: x * y, map_shape) * in_channels)
        else:
            assert type(map_shape) == int
            in_ = in_channels * map_shape * map_shape

        self.fc = nn.Linear(in_, out_channels)
        self.with_norm = with_norm
        self.syncbn = bn_type == 'SyncBN'
        if self.syncbn:
            _, self.bn_output = build_norm_layer(
                dict(type=bn_type), out_channels)
        else:
            self.bn_output = nn.BatchNorm1d(out_channels)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2  # to fit vit model
        x = x[0]
        x = self.bn_input(x)
        x = self.dropout(x)
        x = self.fc(x.view(x.size(0), -1))
        # if self.syncbn:
        x = self._forward_syncbn(self.bn_output, x)
        # else:
        #    x = self.bn_output(x)

        if self.with_norm:
            x = nn.functional.normalize(x, p=2, dim=1)
        return [x]


@NECKS.register_module
class MultiLinearNeck(nn.Module):
    '''MultiLinearNeck neck: MultiFc head
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers=1,
                 with_avg_pool=True):
        """
        Args:
            in_channels: int or list[int]
            out_channels: int or list[int]
            num_layers : total fc num
            with_avg_pool : input will be avgPool if True
        Returns:
            None
        Raises:
            len(in_channel) != len(out_channels)
            len(in_channel) != len(num_layers)
        """
        super(MultiLinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.num_layers = num_layers
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if num_layers == 1:
            self.fc = nn.Linear(in_channels, out_channels)
        else:
            assert len(in_channels) == len(out_channels)
            assert len(in_channels) == num_layers
            self.fc = nn.ModuleList(
                [nn.Linear(i, j) for i, j in zip(in_channels, out_channels)])

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2  # to fit vit model
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = self.fc(x.view(x.size(0), -1))
        return [x]


@NECKS.register_module()
class HRFuseScales(nn.Module):
    """Fuse feature map of multiple scales in HRNet.
    Args:
        in_channels (list[int]): The input channels of all scales.
        out_channels (int): The channels of fused feature map.
            Defaults to 2048.
        norm_cfg (dict): dictionary to construct norm layers.
            Defaults to ``dict(type='BN', momentum=0.1)``.
        init_cfg (dict | list[dict], optional): Initialization config dict.
            Defaults to ``dict(type='Normal', layer='Linear', std=0.01))``.
    """

    def __init__(self,
                 in_channels,
                 out_channels=2048,
                 norm_cfg=dict(type='BN', momentum=0.1)):
        super(HRFuseScales, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm_cfg = norm_cfg

        block_type = Bottleneck
        out_channels = [128, 256, 512, 1024]

        # Increase the channels on each resolution
        # from C, 2C, 4C, 8C to 128, 256, 512, 1024
        increase_layers = []
        for i in range(len(in_channels)):
            increase_layers.append(
                ResLayer(
                    block_type,
                    in_channels=in_channels[i],
                    out_channels=out_channels[i],
                    num_blocks=1,
                    stride=1,
                ))
        self.increase_layers = nn.ModuleList(increase_layers)

        # Downsample feature maps in each scale.
        downsample_layers = []
        for i in range(len(in_channels) - 1):
            downsample_layers.append(
                ConvModule(
                    in_channels=out_channels[i],
                    out_channels=out_channels[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=False,
                ))
        self.downsample_layers = nn.ModuleList(downsample_layers)

        # The final conv block before final classifier linear layer.
        self.final_layer = ConvModule(
            in_channels=out_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
            bias=False,
        )

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == len(self.in_channels)

        feat = self.increase_layers[0](x[0])
        for i in range(len(self.downsample_layers)):
            feat = self.downsample_layers[i](feat) + \
                self.increase_layers[i + 1](x[i + 1])

        return [self.final_layer(feat)]
