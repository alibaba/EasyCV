#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import math
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from easycv.models.utils.profiler import module_profile
from easycv.utils.math_utils import make_divisible
from ...layers import (ConvLayer, Dropout, LinearSelfAttention,
                       get_activation_fn, get_normalization_layer)


class BaseModule(nn.Module):
    """Base class for all modules"""

    def __init__(self, *args, **kwargs):
        super(BaseModule, self).__init__()

    def forward(self, x: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    def profile_module(self, input: Any, *args,
                       **kwargs) -> Tuple[Any, float, float]:
        raise NotImplementedError

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class LinearAttnFFN(BaseModule):
    """
    This class defines the pre-norm transformer encoder with linear self-attention in `MobileViTv2 paper <>`_
    Args:
        opts: command line arguments
        embed_dim (int): :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, P, N)`
        ffn_latent_dim (int): Inner dimension of the FFN
        attn_dropout (Optional[float]): Dropout rate for attention in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers. Default: 0.0
        norm_layer (Optional[str]): Normalization layer. Default: layer_norm_2d

    Shape:
        - Input: :math:`(B, C_{in}, P, N)` where :math:`B` is batch size, :math:`C_{in}` is input embedding dim,
            :math:`P` is number of pixels in a patch, and :math:`N` is number of patches,
        - Output: same shape as the input
    """

    def __init__(self,
                 opts,
                 embed_dim: int,
                 ffn_latent_dim: int,
                 attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.1,
                 ffn_dropout: Optional[float] = 0.0,
                 norm_layer: Optional[str] = 'layer_norm_2d',
                 *args,
                 **kwargs) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            opts, embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True)

        self.pre_norm_attn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim),
            attn_unit,
            Dropout(p=dropout),
        )

        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(
                opts=opts, norm_type=norm_layer, num_features=embed_dim),
            ConvLayer(
                opts=opts,
                in_channels=embed_dim,
                out_channels=ffn_latent_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=True,
            ),
            Dropout(p=ffn_dropout),
            ConvLayer(
                opts=opts,
                in_channels=ffn_latent_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                bias=True,
                use_norm=False,
                use_act=False,
            ),
            Dropout(p=dropout),
        )

        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout
        self.std_dropout = dropout
        self.attn_fn_name = attn_unit.__repr__()
        self.norm_name = norm_layer

    @staticmethod
    def build_act_layer(opts) -> nn.Module:
        act_type = getattr(opts, 'model.activation.name', 'relu')
        neg_slope = getattr(opts, 'model.activation.neg_slope', 0.1)
        inplace = getattr(opts, 'model.activation.inplace', False)
        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    def __repr__(self) -> str:
        return '{}(embed_dim={}, ffn_dim={}, dropout={}, ffn_dropout={}, attn_fn={}, norm_layer={})'.format(
            self.__class__.__name__,
            self.embed_dim,
            self.ffn_dim,
            self.std_dropout,
            self.ffn_dropout,
            self.attn_fn_name,
            self.norm_name,
        )

    def forward(self,
                x: Tensor,
                x_prev: Optional[Tensor] = None,
                *args,
                **kwargs) -> Tensor:
        if x_prev is None:
            # self-attention
            x = x + self.pre_norm_attn(x)
        else:
            # cross-attention
            res = x
            x = self.pre_norm_attn[0](x)  # norm
            x = self.pre_norm_attn[1](x, x_prev)  # attn
            x = self.pre_norm_attn[2](x)  # drop
            x = x + res  # residual

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x

    def profile_module(self, input: Tensor, *args,
                       **kwargs) -> Tuple[Tensor, float, float]:
        out, p_mha, m_mha = module_profile(module=self.pre_norm_attn, x=input)
        out, p_ffn, m_ffn = module_profile(module=self.pre_norm_ffn, x=input)

        macs = m_mha + m_ffn
        params = p_mha + p_ffn

        return input, params, macs


class InvertedResidual(BaseModule):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        opts: command-line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (Optional[int]): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        dilation (Optional[int]): Use conv with dilation. Default: 1
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(self,
                 opts,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: Union[int, float],
                 dilation: int = 1,
                 skip_connection: Optional[bool] = True,
                 *args,
                 **kwargs) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name='exp_1x1',
                module=ConvLayer(
                    opts,
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    use_act=True,
                    use_norm=True,
                ),
            )

        block.add_module(
            name='conv_3x3',
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim,
                use_act=True,
                use_norm=True,
                dilation=dilation,
            ),
        )

        block.add_module(
            name='red_1x1',
            module=ConvLayer(
                opts,
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation
        self.stride = stride
        self.use_res_connect = (
            self.stride == 1 and in_channels == out_channels
            and skip_connection)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

    def profile_module(self, input: Tensor, *args,
                       **kwargs) -> Tuple[Tensor, float, float]:
        return module_profile(module=self.block, x=input)

    def __repr__(self) -> str:
        return '{}(in_channels={}, out_channels={}, stride={}, exp={}, dilation={}, skip_conn={})'.format(
            self.__class__.__name__,
            self.in_channels,
            self.out_channels,
            self.stride,
            self.exp,
            self.dilation,
            self.use_res_connect,
        )


class MobileViTBlockv2(BaseModule):
    """
    This class defines the `MobileViTv2 block <>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(self,
                 opts,
                 in_channels: int,
                 attn_unit_dim: int,
                 ffn_multiplier: Optional[Union[Sequence[Union[int, float]],
                                                int, float]] = 2.0,
                 n_attn_blocks: Optional[int] = 2,
                 attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.0,
                 ffn_dropout: Optional[float] = 0.0,
                 patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8,
                 conv_ksize: Optional[int] = 3,
                 dilation: Optional[int] = 1,
                 attn_norm_layer: Optional[str] = 'layer_norm_2d',
                 *args,
                 **kwargs) -> None:
        cnn_out_dim = attn_unit_dim

        conv_3x3_in = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            dilation=dilation,
            groups=in_channels,
        )
        conv_1x1_in = ConvLayer(
            opts=opts,
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )

        super(MobileViTBlockv2, self).__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            opts=opts,
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.conv_proj = ConvLayer(
            opts=opts,
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize
        self.enable_coreml_compatible_fn = getattr(
            opts, 'common.enable_coreml_compatible_module', False)

        if self.enable_coreml_compatible_fn:
            # we set persistent to false so that these weights are not part of model's state_dict
            self.register_buffer(
                name='unfolding_weights',
                tensor=self._compute_unfolding_weights(),
                persistent=False,
            )

        self.activate_fn = nn.Softmax(dim=1)

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w))
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights

    def _build_attn_layer(self, opts, d_model: int, ffn_mult: Union[Sequence,
                                                                    int,
                                                                    float],
                          n_layers: int, attn_dropout: float, dropout: float,
                          ffn_dropout: float, attn_norm_layer: str, *args,
                          **kwargs) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) *
                d_model)
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                opts=opts,
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            ) for block_idx in range(n_layers)
        ]
        global_rep.append(
            get_normalization_layer(
                opts=opts, norm_type=attn_norm_layer, num_features=d_model))

        return nn.Sequential(*global_rep), d_model

    def __repr__(self) -> str:
        repr_str = '{}('.format(self.__class__.__name__)

        repr_str += '\n\t Local representations'
        if isinstance(self.local_rep, nn.Sequential):
            for m in self.local_rep:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.local_rep)

        repr_str += '\n\t Global representations with patch size of {}x{}'.format(
            self.patch_h,
            self.patch_w,
        )
        if isinstance(self.global_rep, nn.Sequential):
            for m in self.global_rep:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.global_rep)

        if isinstance(self.conv_proj, nn.Sequential):
            for m in self.conv_proj:
                repr_str += '\n\t\t {}'.format(m)
        else:
            repr_str += '\n\t\t {}'.format(self.conv_proj)

        repr_str += '\n)'
        return repr_str

    def unfolding_pytorch(
            self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(batch_size, in_channels,
                                  self.patch_h * self.patch_w, -1)

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor,
                        output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def unfolding_coreml(
            self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        # im2col is not implemented in Coreml, so here we hack its implementation using conv2d
        # we compute the weights

        # [B, C, H, W] --> [B, C, P, N]
        batch_size, in_channels, img_h, img_w = feature_map.shape
        #
        patches = F.conv2d(
            feature_map,
            self.unfolding_weights,
            bias=None,
            stride=(self.patch_h, self.patch_w),
            padding=0,
            dilation=1,
            groups=in_channels,
        )
        patches = patches.reshape(batch_size, in_channels,
                                  self.patch_h * self.patch_w, -1)
        return patches, (img_h, img_w)

    def folding_coreml(self, patches: Tensor,
                       output_size: Tuple[int, int]) -> Tensor:
        # col2im is not supported on coreml, so tracing fails
        # We hack folding function via pixel_shuffle to enable coreml tracing
        batch_size, in_dim, patch_size, n_patches = patches.shape

        n_patches_h = output_size[0] // self.patch_h
        n_patches_w = output_size[1] // self.patch_w

        feature_map = patches.reshape(batch_size,
                                      in_dim * self.patch_h * self.patch_w,
                                      n_patches_h, n_patches_w)
        assert (self.patch_h == self.patch_w
                ), 'For Coreml, we need patch_h and patch_w are the same'
        feature_map = F.pixel_shuffle(feature_map, upscale_factor=self.patch_h)
        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode='bilinear', align_corners=True)
        return x

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm)
        else:
            patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations on all patches
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm

    def forward_temporal(self, x: Tensor, x_prev: Tensor, *args,
                         **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        # convert feature map to patches
        if self.enable_coreml_compatible_fn:
            patches, output_size = self.unfolding_coreml(fm)
        else:
            patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations
        for global_layer in self.global_rep:
            if isinstance(global_layer, LinearAttnFFN):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        if self.enable_coreml_compatible_fn:
            fm = self.folding_coreml(patches=patches, output_size=output_size)
        else:
            fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm, patches

    def forward(
            # self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
            self,
            img,
            *args,
            **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(img, Tuple) and len(img) == 2:
            # for spatio-temporal data (e.g., videos)
            return self.forward_temporal(x=img[0], x_prev=img[1])
        # elif isinstance(img, Tensor):
        # for image data
        #    return self.forward_spatial(img)
        else:
            out = self.forward_spatial(x=img, **kwargs)
            return out
            # raise NotImplementedError

    def profile_module(self, input: Tensor, *args,
                       **kwargs) -> Tuple[Tensor, float, float]:
        params = macs = 0.0
        input = self.resize_input_if_needed(input)

        res = input
        out, p, m = module_profile(module=self.local_rep, x=input)
        params += p
        macs += m

        patches, output_size = self.unfolding_pytorch(feature_map=out)

        patches, p, m = module_profile(module=self.global_rep, x=patches)
        params += p
        macs += m

        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        out, p, m = module_profile(module=self.conv_proj, x=fm)
        params += p
        macs += m

        return res, params, macs
