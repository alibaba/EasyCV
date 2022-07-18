# Copyright (c) Alibaba, Inc. and its affiliates.
# Reference: https://github.com/Alpha-VL/FastConvMAE
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

from easycv.models.registry import BACKBONES
from easycv.models.utils.pos_embed import get_2d_sincos_pos_embed
from .vit_transfomer_dynamic import Block, DropPath


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding.
    Args:
        img_size (int | tuple): The size of input image
        patch_size (int | tiple): The size of one patch
        in_channels (int): The num of input channels. Default: 3
        embed_dims (int): The dimensions of embedding. Default: 768
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768):
        super().__init__()
        if not isinstance(img_size, (list, tuple)):
            img_size = (img_size, img_size)
        if not isinstance(patch_size, (list, tuple)):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0],
                          img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.GELU()

    def forward(self, x):
        _, _, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.act(x)


class ConvMlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ConvMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            residual = x
            x = self.conv1(
                self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            x1 = self.attn(mask[0] * x)
            x2 = self.attn(mask[1] * x)
            x3 = self.attn(mask[2] * x)
            x4 = self.attn(mask[3] * x)
            x = mask[0] * x1 + mask[1] * x2 + mask[2] * x3 + mask[3] * x4
            x = residual + self.drop_path(self.conv2(x))
        else:
            x = x + self.drop_path(
                self.conv2(
                    self.attn(
                        self.conv1(
                            self.norm1(x.permute(0, 2, 3, 1)).permute(
                                0, 3, 1, 2)))))
        x = x + self.drop_path(
            self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x


@BACKBONES.register_module
class FastConvMAEViT(nn.Module):
    """ Fast ConvMAE framework is a superiorly fast masked modeling scheme via
    complementary masking and mixture of reconstrunctors based on the ConvMAE(https://arxiv.org/abs/2205.03892).

    Args:
        img_size (list | tuple): Input image size for three stages.
        patch_size (list | tuple): The patch size for three stages.
        in_channels (int): The num of input channels. Default: 3
        embed_dim (list | tuple): The dimensions of embedding for three stages.
        depth (list | tuple): depth for three stages.
        num_heads (int): Parallel attention heads
        mlp_ratio (list | tuple): Mlp expansion ratio.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Defaults to 0.
        drop_path_rate (float): Stochastic depth rate. Defaults to 0.
        norm_layer (nn.Module): normalization layer
        init_pos_embed_by_sincos: initialize pos_embed by sincos strategy
        with_fuse(bool): Whether to use fuse layers.
        global_pool: global pool

    """

    def __init__(
        self,
        img_size=[224, 56, 28],
        patch_size=[4, 2, 2],
        in_channels=3,
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        drop_rate=0.,
        drop_path_rate=0.1,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_pos_embed_by_sincos=True,
        with_fuse=True,
        global_pool=False,
    ):

        super().__init__()
        self.init_pos_embed_by_sincos = init_pos_embed_by_sincos
        self.with_fuse = with_fuse
        self.global_pool = global_pool

        assert len(img_size) == len(patch_size) == len(embed_dim) == len(
            mlp_ratio)

        self.patch_size = patch_size[0] * patch_size[1] * patch_size[2]
        self.patch_embed1 = PatchEmbed(
            img_size=img_size[0],
            patch_size=patch_size[0],
            in_channels=in_channels,
            embed_dim=embed_dim[0])
        self.patch_embed2 = PatchEmbed(
            img_size=img_size[1],
            patch_size=patch_size[1],
            in_channels=embed_dim[0],
            embed_dim=embed_dim[1])
        self.patch_embed3 = PatchEmbed(
            img_size=img_size[2],
            patch_size=patch_size[2],
            in_channels=embed_dim[1],
            embed_dim=embed_dim[2])

        self.patch_embed4 = nn.Linear(embed_dim[2], embed_dim[2])

        if with_fuse:
            self._make_fuse_layers(embed_dim)

        self.num_patches = self.patch_embed3.num_patches
        if init_pos_embed_by_sincos:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim[2]),
                requires_grad=False)
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_patches, embed_dim[2]), )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, sum(depth))

        self.blocks1 = nn.ModuleList([
            ConvBlock(
                dim=embed_dim[0],
                mlp_ratio=mlp_ratio[0],
                drop=drop_rate,
                drop_path=dpr[i],
            ) for i in range(depth[0])
        ])
        self.blocks2 = nn.ModuleList([
            ConvBlock(
                dim=embed_dim[1],
                mlp_ratio=mlp_ratio[1],
                drop=drop_rate,
                drop_path=dpr[depth[0] + i],
            ) for i in range(depth[1])
        ])
        self.blocks3 = nn.ModuleList([
            Block(
                dim=embed_dim[2],
                num_heads=num_heads,
                mlp_ratio=mlp_ratio[2],
                qkv_bias=True,
                qk_scale=None,
                drop=drop_rate,
                drop_path=dpr[depth[0] + depth[1] + i],
                norm_layer=norm_layer) for i in range(depth[2])
        ])

        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim[-1])
            self.norm = None
        else:
            self.norm = norm_layer(embed_dim[-1])
            self.fc_norm = None

    def init_weights(self):
        if self.init_pos_embed_by_sincos:
            # initialize (and freeze) pos_embed by sin-cos embedding
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.num_patches**.5),
                cls_token=False)
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))
        else:
            trunc_normal_(self.pos_embed, std=.02)

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed3.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _make_fuse_layers(self, embed_dim):
        self.stage1_output_decode = nn.Conv2d(
            embed_dim[0], embed_dim[2], 4, stride=4)
        self.stage2_output_decode = nn.Conv2d(
            embed_dim[1], embed_dim[2], 2, stride=2)

    def random_masking(self, x, mask_ratio=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N = x.shape[0]
        L = self.num_patches
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep1 = ids_shuffle[:, :len_keep]
        ids_keep2 = ids_shuffle[:, len_keep:2 * len_keep]
        ids_keep3 = ids_shuffle[:, 2 * len_keep:3 * len_keep]
        ids_keep4 = ids_shuffle[:, 3 * len_keep:]

        # generate the binary mask: 0 is keep, 1 is remove
        mask1 = torch.ones([N, L], device=x.device)
        mask1[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask1 = torch.gather(mask1, dim=1, index=ids_restore)

        mask2 = torch.ones([N, L], device=x.device)
        mask2[:, len_keep:2 * len_keep] = 0
        # unshuffle to get the binary mask
        mask2 = torch.gather(mask2, dim=1, index=ids_restore)

        mask3 = torch.ones([N, L], device=x.device)
        mask3[:, 2 * len_keep:3 * len_keep] = 0
        # unshuffle to get the binary mask
        mask3 = torch.gather(mask3, dim=1, index=ids_restore)

        mask4 = torch.ones([N, L], device=x.device)
        mask4[:, 3 * len_keep:4 * len_keep] = 0
        # unshuffle to get the binary mask
        mask4 = torch.gather(mask4, dim=1, index=ids_restore)

        return [ids_keep1, ids_keep2, ids_keep3,
                ids_keep4], [mask1, mask2, mask3, mask4], ids_restore

    def _fuse_forward(self, s1, s2, ids_keep=None, mask_ratio=None):
        stage1_embed = self.stage1_output_decode(s1).flatten(2).permute(
            0, 2, 1)
        stage2_embed = self.stage2_output_decode(s2).flatten(2).permute(
            0, 2, 1)

        if mask_ratio is not None:
            stage1_embed_1 = torch.gather(
                stage1_embed,
                dim=1,
                index=ids_keep[0].unsqueeze(-1).repeat(1, 1,
                                                       stage1_embed.shape[-1]))
            stage2_embed_1 = torch.gather(
                stage2_embed,
                dim=1,
                index=ids_keep[0].unsqueeze(-1).repeat(1, 1,
                                                       stage2_embed.shape[-1]))
            stage1_embed_2 = torch.gather(
                stage1_embed,
                dim=1,
                index=ids_keep[1].unsqueeze(-1).repeat(1, 1,
                                                       stage1_embed.shape[-1]))
            stage2_embed_2 = torch.gather(
                stage2_embed,
                dim=1,
                index=ids_keep[1].unsqueeze(-1).repeat(1, 1,
                                                       stage2_embed.shape[-1]))
            stage1_embed_3 = torch.gather(
                stage1_embed,
                dim=1,
                index=ids_keep[2].unsqueeze(-1).repeat(1, 1,
                                                       stage1_embed.shape[-1]))
            stage2_embed_3 = torch.gather(
                stage2_embed,
                dim=1,
                index=ids_keep[2].unsqueeze(-1).repeat(1, 1,
                                                       stage2_embed.shape[-1]))
            stage1_embed_4 = torch.gather(
                stage1_embed,
                dim=1,
                index=ids_keep[3].unsqueeze(-1).repeat(1, 1,
                                                       stage1_embed.shape[-1]))
            stage2_embed_4 = torch.gather(
                stage2_embed,
                dim=1,
                index=ids_keep[3].unsqueeze(-1).repeat(1, 1,
                                                       stage2_embed.shape[-1]))

            stage1_embed = torch.cat([
                stage1_embed_1, stage1_embed_2, stage1_embed_3, stage1_embed_4
            ])
            stage2_embed = torch.cat([
                stage2_embed_1, stage2_embed_2, stage2_embed_3, stage2_embed_4
            ])

        return stage1_embed, stage2_embed

    def forward(self, x, mask_ratio=None):
        if mask_ratio is not None:
            assert self.with_fuse

        # embed patches
        if mask_ratio is not None:
            ids_keep, masks, ids_restore = self.random_masking(x, mask_ratio)
            mask_for_patch1 = [
                1 - mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(
                    1, 1, 1, 16).reshape(-1, 14, 14, 4, 4).permute(
                        0, 1, 3, 2, 4).reshape(x.shape[0], 56, 56).unsqueeze(1)
                for mask in masks
            ]
            mask_for_patch2 = [
                1 - mask.reshape(-1, 14, 14).unsqueeze(-1).repeat(
                    1, 1, 1, 4).reshape(-1, 14, 14, 2, 2).permute(
                        0, 1, 3, 2, 4).reshape(x.shape[0], 28, 28).unsqueeze(1)
                for mask in masks
            ]
        else:
            mask_for_patch1 = None
            mask_for_patch2 = None

        s1 = self.patch_embed1(x)
        s1 = self.pos_drop(s1)
        for blk in self.blocks1:
            s1 = blk(s1, mask_for_patch1)

        s2 = self.patch_embed2(s1)
        for blk in self.blocks2:
            s2 = blk(s2, mask_for_patch2)

        if self.with_fuse:
            stage1_embed, stage2_embed = self._fuse_forward(
                s1, s2, ids_keep, mask_ratio)

        x = self.patch_embed3(s2)
        x = x.flatten(2).permute(0, 2, 1)
        x = self.patch_embed4(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed

        if mask_ratio is not None:
            x1 = torch.gather(
                x,
                dim=1,
                index=ids_keep[0].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x2 = torch.gather(
                x,
                dim=1,
                index=ids_keep[1].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x3 = torch.gather(
                x,
                dim=1,
                index=ids_keep[2].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x4 = torch.gather(
                x,
                dim=1,
                index=ids_keep[3].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
            x = torch.cat([x1, x2, x3, x4])

        # apply Transformer blocks
        for blk in self.blocks3:
            x = blk(x)

        if self.with_fuse:
            x = x + stage1_embed + stage2_embed

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            x = self.fc_norm(x)
        else:
            x = self.norm(x)

        if mask_ratio is not None:
            mask = torch.cat([masks[0], masks[1], masks[2], masks[3]])
            return x, mask, ids_restore

        return x, None, None
