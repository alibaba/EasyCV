# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from easycv.models.utils import DropPath, Mlp
from ..registry import BACKBONES


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 use_layer_scale=False,
                 init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.gamma_1 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)
            self.gamma_2 = nn.Parameter(
                init_values * torch.ones((dim)), requires_grad=True)

    def forward(self, x, return_attention=False, rel_pos_bias=None):
        y, attn = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
        if return_attention:
            return attn
        if self.use_layer_scale:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_fea_and_attn(self, x):
        y, attn = self.attn(self.norm1(x))
        if self.use_layer_scale:
            x = x + self.drop_path(self.gamma_1 * y)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(y)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module
class VisionTransformer(nn.Module):
    """ DeiT III is based on ViT. It uses some strategies to make the vit model
    better, just like layer scale, stochastic depth, 3-Augment.

    Paper link: https://arxiv.org/pdf/2204.07118.pdf (DeiT III: Revenge of the ViT)

    Args:
        img_size (list): Input image size. img_size=[224] means the image size is
            224*224. img_size=[192, 224] means the image size is 192*224.
        patch_size (int): The patch size. Default: 16
        in_chans (int): The num of input channels. Default: 3
        num_classes (int): The num of picture classes. Default: 1000
        embed_dim (int): The dimensions of embedding. Default: 768
        depth (int): The num of blocks. Default: 12
        num_heads (int): Parallel attention heads. Default: 12
        mlp_ratio (float): Mlp expansion ratio. Default: 4.0
        qkv_bias (bool): Does kqv use bias. Default: False
        qk_scale (float | None): In the step of self-attention, if qk_scale is not
            None, it will use qk_scale to scale the q @ k. Otherwise it will use
            head_dim**-0.5 instead of qk_scale. Default: None
        drop_rate (float): Probability of an element to be zeroed after the feed
            forward layer. Default: 0.0
        drop_path_rate (float): Stochastic depth rate. Default: 0
        norm_layer (nn.Module): normalization layer
        use_dense_prediction (bool): If use_dense_prediction is True, the global
            pool and norm will before head will be removed.(if any) Default: False
        global_pool (bool): Global pool before head. Default: False
        use_layer_scale (bool): If use_layer_scale is True, it will use layer
            scale. Default: False
        init_scale (float): It is used for layer scale in Block to scale the
            gamma_1 and gamma_2.

    """

    def __init__(self,
                 img_size=[224],
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 use_dense_prediction=False,
                 global_pool=False,
                 use_layer_scale=False,
                 init_scale=1e-4,
                 **kwargs):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.drop_path_rate = drop_path_rate
        self.depth = depth
        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                use_layer_scale=use_layer_scale,
                init_values=init_scale) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(
            embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Dense prediction head
        self.use_dense_prediction = use_dense_prediction
        if self.use_dense_prediction:
            self.head_dense = None

        # Use global average pooling
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            self.norm = None

    def init_weights(self):
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):

        x = self.forward_features(x)
        x = self.pos_drop(x)
        x = self.head(x)

        return [x]

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = x + self.pos_embed
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)

        if self.use_dense_prediction:
            return x[:, 0], x[:, 1:]
        else:
            if self.global_pool:
                x = x[:, 1:, :].mean(dim=1)
                return self.fc_norm(x)
            else:
                return x[:, 0]