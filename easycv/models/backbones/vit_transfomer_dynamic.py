# Copyright (c) Alibaba, Inc. and its affiliates.
"""
Mostly copy-paste from timm library.
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

dynamic Input support borrow from
https://github.com/microsoft/esvit/blob/main/models/vision_transformer.py

"""
import math
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from easycv.models.utils import DropPath, Mlp


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
                 norm_layer=nn.LayerNorm):
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

    def forward(self, x, return_attention=False, rel_pos_bias=None):
        y, attn = self.attn(self.norm1(x), rel_pos_bias=rel_pos_bias)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def forward_fea_and_attn(self, x):
        y, attn = self.attn(self.norm1(x))
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


class DynamicVisionTransformer(nn.Module):
    """Dynamic Vision Transformer """

    def __init__(self,
                 img_size=[224],
                 patch_size=16,
                 in_chans=3,
                 num_classes=0,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_dense_prediction=False,
                 global_pool=False,
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
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
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
                norm_layer=norm_layer) for i in range(depth)
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

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        # Perform forward pass separately on each resolution input.
        # The inputs corresponding to a single resolution are clubbed and single
        # forward is run on the same resolution inputs. Hence we do several
        # forward passes = number of different resolutions used. We then
        # concatenate all the output features.
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in x]),
                return_counts=True,
            )[1], 0)

        if self.use_dense_prediction:
            start_idx = 0

            for end_idx in idx_crops:
                _out_cls, _out_fea = self.forward_features(
                    torch.cat(x[start_idx:end_idx]))
                B, N, C = _out_fea.shape

                if start_idx == 0:
                    output_cls = _out_cls
                    output_fea = _out_fea.reshape(B * N, C)
                    npatch = [N]
                else:
                    output_cls = torch.cat((output_cls, _out_cls))
                    output_fea = torch.cat(
                        (output_fea, _out_fea.reshape(B * N, C)))
                    npatch.append(N)
                start_idx = end_idx

            return [
                self.head(output_cls),
                self.head_dense(output_fea), output_fea, npatch
            ]

        else:
            start_idx = 0
            for end_idx in idx_crops:
                _out = self.forward_features(torch.cat(x[start_idx:end_idx]))
                # _out = self.forward_return_n_last_blocks(torch.cat(x[start_idx: end_idx]), 4, True)
                if start_idx == 0:
                    output = _out
                else:
                    output = torch.cat((output, _out))
                start_idx = end_idx

            # print(f'output[0] {output[0].shape}')
            # Run the head forward on the concatenated features.
            return [self.head(output)]

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)

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

    def forward_feature_maps(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        if self.norm is not None:
            x = self.norm(x)

        return x

    def interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1] - 1
        N = pos_embed.shape[1] - 1
        if npatch == N:
            return pos_embed
        class_emb = pos_embed[:, 0]
        pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                              dim).permute(0, 3, 1, 2),
            scale_factor=math.sqrt(npatch / N),
            mode='bicubic',
        )
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_emb.unsqueeze(0), pos_embed), dim=1)

    def forward_selfattention(self, x, n=1):
        # n=1 return the last layer attn map; otherwise return attn maps in all layers

        B, nc, w, h = x.shape
        N = self.pos_embed.shape[1] - 1
        x = self.patch_embed(x)

        # interpolate patch embeddings
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                    dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        if w0 != patch_pos_embed.shape[-2]:
            helper = torch.zeros(h0)[None, None, None, :].repeat(
                1, dim, w0 - patch_pos_embed.shape[-2], 1).to(x.device)
            patch_pos_embed = torch.cat((patch_pos_embed, helper), dim=-2)
        if h0 != patch_pos_embed.shape[-1]:
            helper = torch.zeros(w0)[None, None, :, None].repeat(
                1, dim, 1, h0 - patch_pos_embed.shape[-1]).to(x.device)
            pos_embed = torch.cat((patch_pos_embed, helper), dim=-1)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        pos_embed = torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed),
                              dim=1)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + pos_embed
        x = self.pos_drop(x)

        if n == 1:
            return self.forward_last_selfattention(x)
        else:
            return self.forward_all_selfattention(x)

    def forward_last_selfattention(self, x):
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def forward_all_selfattention(self, x):
        attn_out = []
        for i, blk in enumerate(self.blocks):
            x, attn = blk.forward_fea_and_attn(x)
            attn_out.append(attn)

        return attn_out

    def forward_return_n_last_blocks(self,
                                     x,
                                     n=1,
                                     return_patch_avgpool=False,
                                     depths=[]):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self.interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed
        x = self.pos_drop(x)

        # we will return the [CLS] tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x)[:, 0])
        if return_patch_avgpool:
            x = self.norm(x)
            # In addition to the [CLS] tokens from the `n` last blocks, we also return
            # the patch tokens from the last block. This is useful for linear eval.
            output.append(torch.mean(x[:, 1:], dim=1))
        return torch.cat(output, dim=-1)


def dynamic_deit_tiny_p16(patch_size=16, **kwargs):
    model = DynamicVisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def dynamic_deit_small_p16(patch_size=16, **kwargs):
    model = DynamicVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def dynamic_vit_base_p16(patch_size=16, **kwargs):
    model = DynamicVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def dynamic_vit_large_p16(patch_size=16, **kwargs):
    model = DynamicVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model


def dynamic_vit_huge_p14(patch_size=14, **kwargs):
    model = DynamicVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    return model
