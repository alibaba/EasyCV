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

from easycv.models.backbones.vision_transformer import Block, VisionTransformer


class DynamicVisionTransformer(VisionTransformer):
    """Dynamic Vision Transformer

    Args:
        use_dense_prediction (bool): If use_dense_prediction is True, the global
            pool and norm will before head will be removed.(if any) Default: False

    """

    def __init__(self, use_dense_prediction=False, **kwargs):
        super(DynamicVisionTransformer, self).__init__(**kwargs)

        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.embed_dim))

        if self.hydra_attention:
            hy = [
                x >= (self.depth - self.hydra_attention_layers)
                for x in range(self.depth)
            ]
            head = [
                self.embed_dim if x >=
                (self.depth - self.hydra_attention_layers) else self.num_heads
                for x in range(self.depth)
            ]
        else:
            hy = [False for x in range(self.depth)]
            head = [self.num_heads for x in range(self.depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim,
                num_heads=head[i],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.dpr[i],
                norm_layer=self.norm_layer,
                use_layer_scale=self.use_layer_scale,
                init_values=self.init_scale,
                hydra_attention=hy[i]) for i in range(self.depth)
        ])

        # Dense prediction head
        self.use_dense_prediction = use_dense_prediction
        if self.use_dense_prediction:
            self.head_dense = None

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
