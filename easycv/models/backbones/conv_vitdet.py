# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import variance_scaling_

from easycv.file import io, is_oss_path
from easycv.utils.constant import CACHE_DIR
from ..registry import BACKBONES
from .conv_mae_vit import FastConvMAEViT


def trunc_normal_(tensor, mean=0.0, std=1.0):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class RelativePositionBias(nn.Module):

    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] -
                                      1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(self.num_relative_distance,
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (coords_flatten[:, :, None] -
                           coords_flatten[:, None, :])  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer('relative_position_index',
                             relative_position_index)

    def forward(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1,
            )  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger = logging.getLogger(__name__)
    _logger.info('Resized position embedding: %s to %s', posemb.shape,
                 posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0,
                                                                 num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s',
                 [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old,
                                      -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(
        posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3,
                                      1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


@BACKBONES.register_module()
class ConvViTDet(FastConvMAEViT):
    """Reference: https://github.com/Alpha-VL/FastConvMAE
    Args:
        window_size (int): The height and width of the window.
        in_channels (int): The num of input channels. Default: 3
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        img_size (list | tuple): Input image size for three stages.
        embed_dim (list | tuple): The dimensions of embedding for three stages.
        patch_size (list | tuple): The patch size for three stages.
        depth (list | tuple): depth for three stages.
        num_heads (int): Parallel attention heads
        mlp_ratio (list | tuple): Mlp expansion ratio.
        norm_layer (nn.Module): normalization layer
        init_pos_embed_by_sincos: initialize pos_embed by sincos strategy
        pretrained (str): pretrained path
    """

    def __init__(
        self,
        window_size,
        in_channels=3,
        drop_rate=0.0,
        drop_path_rate=0.0,
        img_size=[1024, 256, 128],
        embed_dim=[256, 384, 768],
        patch_size=[4, 2, 2],
        depth=[2, 2, 11],
        mlp_ratio=[4, 4, 4],
        num_heads=12,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_pos_embed_by_sincos=False,
        pretrained=None,
    ):
        super(ConvViTDet, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_pos_embed_by_sincos=init_pos_embed_by_sincos,
            with_fuse=False,
            global_pool=False,
        )

        self.pretrained = pretrained
        self.num_heads = num_heads
        self.num_features = (self.embed_dim) = embed_dim[
            -1]  # num_features for consistency with other models
        self.num_tokens = None
        self.norm = None

        self.window_size = window_size
        self.ms_adaptor = nn.ModuleList(
            [nn.Identity(),
             nn.Identity(),
             nn.Identity(),
             nn.MaxPool2d(2)])
        self.ms_adaptor.apply(self.init_adaptor)

        self.windowed_rel_pos_bias = RelativePositionBias(
            window_size=(self.window_size, self.window_size),
            num_heads=self.num_heads)
        self.global_rel_pos_bias = RelativePositionBias(
            window_size=self.patch_embed3.grid_size, num_heads=self.num_heads)

        # self._out_features = ["s0", "s1", "s2", "s3"]
        # self._out_feature_channels = {"s0": 256, "s1": 384, "s2": 768, "s3": 768}
        # self._out_feature_strides = {"s0": 4, "s1": 8, "s2": 16, "s3": 32}
        # self.global_attn_index = [0, 4, 8, 10]

    def init_weights(self):
        super().init_weights()
        if self.pretrained is not None:
            logging.info(f'Load pretrained model from {self.pretrained}...')
            self.load_pretrained(self.pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_adaptor(self, m):
        if isinstance(m, nn.Conv2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose2d):
            lecun_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, images):
        outs = dict()
        x = images
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        outs['s0'] = x
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        outs['s1'] = x
        x = self.patch_embed3(x)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        x = x + self.pos_embed

        x = self.blocks3[0](x, rel_pos_bias=self.global_rel_pos_bias())
        x = rearrange(
            x,
            'b (h w) c -> b h w c',
            h=self.patch_embed3.grid_size[0],
            w=self.patch_embed3.grid_size[1],
        )
        x = rearrange(
            x,
            'b (h h1) (w w1) c -> (b h w) (h1 w1) c',
            h1=self.window_size,
            w1=self.window_size,
        )
        for blk in self.blocks3[1:3]:
            x = blk(x, rel_pos_bias=self.windowed_rel_pos_bias())

        x = rearrange(
            x,
            '(b h w) (h1 w1) c -> b (h h1 w w1) c',
            h=self.patch_embed3.grid_size[0] // self.window_size,
            w=self.patch_embed3.grid_size[1] // self.window_size,
            h1=self.window_size,
            w1=self.window_size,
        )
        x = self.blocks3[3](x, rel_pos_bias=self.global_rel_pos_bias())
        x = rearrange(
            x,
            'b (h w) c -> b h w c',
            h=self.patch_embed3.grid_size[0],
            w=self.patch_embed3.grid_size[1],
        )
        x = rearrange(
            x,
            'b (h h1) (w w1) c -> (b h w) (h1 w1) c',
            h1=self.window_size,
            w1=self.window_size,
        )
        for blk in self.blocks3[4:6]:
            x = blk(x, rel_pos_bias=self.windowed_rel_pos_bias())

        x = rearrange(
            x,
            '(b h w) (h1 w1) c -> b (h h1 w w1) c',
            h=self.patch_embed3.grid_size[0] // self.window_size,
            w=self.patch_embed3.grid_size[1] // self.window_size,
            h1=self.window_size,
            w1=self.window_size,
        )
        x = self.blocks3[6](x, rel_pos_bias=self.global_rel_pos_bias())
        x = rearrange(
            x,
            'b (h w) c -> b h w c',
            h=self.patch_embed3.grid_size[0],
            w=self.patch_embed3.grid_size[1],
        )
        x = rearrange(
            x,
            'b (h h1) (w w1) c -> (b h w) (h1 w1) c',
            h1=self.window_size,
            w1=self.window_size,
        )
        for blk in self.blocks3[7:10]:
            x = blk(x, rel_pos_bias=self.windowed_rel_pos_bias())

        x = rearrange(
            x,
            '(b h w) (h1 w1) c -> b (h h1 w w1) c',
            h=self.patch_embed3.grid_size[0] // self.window_size,
            w=self.patch_embed3.grid_size[1] // self.window_size,
            h1=self.window_size,
            w1=self.window_size,
        )
        x = self.blocks3[10](x, rel_pos_bias=self.global_rel_pos_bias())
        x = rearrange(
            x,
            'b (h w) c -> b c h w',
            h=self.patch_embed3.grid_size[0],
            w=self.patch_embed3.grid_size[1],
        )

        outs['s2'] = x
        outs['s3'] = self.ms_adaptor[-1](x)

        return [outs['s0'], outs['s1'], outs['s2'], outs['s3']]

    def load_pretrained(self, pretrained):
        from mmcv.runner.checkpoint import _load_checkpoint

        if is_oss_path(pretrained):
            _, fname = os.path.split(pretrained)
            cache_file = os.path.join(CACHE_DIR, fname)
            if not os.path.exists(cache_file):
                print(f'download checkpoint from {pretrained} to {cache_file}')
                io.copy(pretrained, cache_file)
            if torch.distributed.is_available(
            ) and torch.distributed.is_initialized():
                torch.distributed.barrier()
            pretrained = cache_file

        unexpected_keys, missing_keys = [], []
        checkpoint = _load_checkpoint(pretrained, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        else:
            checkpoint = checkpoint

        for k in self.state_dict().keys():
            if k not in checkpoint.keys():
                missing_keys.append(k)
        for k in checkpoint.keys():
            if k not in self.state_dict().keys():
                unexpected_keys.append(k)

        if 'pos_embed' in checkpoint:
            checkpoint['pos_embed'] = resize_pos_embed(
                checkpoint['pos_embed'],
                self.pos_embed,
                self.num_tokens,
                self.patch_embed3.grid_size,
            )
        self.load_state_dict(checkpoint, strict=False)
        print(f'Loading ViT pretrained weights from {pretrained}.')
        print(f'missing keys: {missing_keys}')
        print(f'unexpected keys: {unexpected_keys}')

        if 'rel_pos_bias.relative_position_bias_table' in unexpected_keys:
            windowed_relative_position_bias_table = resize_pos_embed(
                checkpoint['rel_pos_bias.relative_position_bias_table'][
                    None, :-3],
                self.windowed_rel_pos_bias.relative_position_bias_table[None],
                0,
            )
            global_relative_position_bias_table = resize_pos_embed(
                checkpoint['rel_pos_bias.relative_position_bias_table'][
                    None, :-3],
                self.global_rel_pos_bias.relative_position_bias_table[None],
                0,
            )
            self.windowed_rel_pos_bias.load_state_dict(
                {
                    'relative_position_bias_table':
                    windowed_relative_position_bias_table[0]
                },
                strict=False)
            self.global_rel_pos_bias.load_state_dict(
                {
                    'relative_position_bias_table':
                    global_relative_position_bias_table[0]
                },
                strict=False)
            print('Load positional bias table from pretrained.')
