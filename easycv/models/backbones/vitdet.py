# Copyright (c) OpenMMLab. All rights reserved.
from typing import Sequence

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer
from mmcv.cnn.utils.weight_init import trunc_normal_
from mmcv.runner.base_module import BaseModule, ModuleList
from mmcv.utils import to_2tuple

from ...utils import get_root_logger
from ..builder import BACKBONES

def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, window_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, interpolate_mode='bicubic'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.interpolate_mode = interpolate_mode

        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't " \
            f'match model ({self.img_size[0]}*{self.img_size[1]}).'
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        #x = self.norm(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        #self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        #x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=None, use_rel_pos_bias=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.softmax = nn.Softmax(dim=-1)
        
        # set rel_pos_bias
        self.use_rel_pos_bias = use_rel_pos_bias
        self.window_size = window_size
        if use_rel_pos_bias:
          rel_sp_dim = 2 * window_size[0] - 1
          self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
          self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.use_rel_pos_bias:
            attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)
        #attn = attn.softmax(dim=-1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, window_size=None, use_rel_pos_bias=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, window_size=window_size, use_rel_pos_bias=use_rel_pos_bias)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

@BACKBONES.register_module()
class ViTDet(BaseModule):
    """Vision Transformer.

    A PyTorch implement of : `An Image is Worth 16x16 Words: Transformers
    for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_

    Args:
        arch (str | dict): Vision Transformer architecture
            Default: 'b'
        img_size (int | tuple): Input image size
        patch_size (int | tuple): The patch size
        out_indices (Sequence | int): Output from which stages.
            Defaults to -1, means the last stage.
        drop_rate (float): Probability of an element to be zeroed.
            Defaults to 0.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='LN')``.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Defaults to True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Defaults to True.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Defaults to "bicubic".
        patch_cfg (dict): Configs of patch embeding. Defaults to an empty dict.
        layer_cfgs (Sequence | dict): Configs of each transformer layer in
            encoder. Defaults to an empty dict.
        init_cfg (dict, optional): Initialization config dict.
            Defaults to None.
    """
    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
                'qkv_bias': False
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
    }
    # Some structures have multiple extra tokens, like DeiT.
    num_extra_tokens = 1  # cls_token

    def __init__(self,
                 arch='b',
                 img_size=224,
                 patch_size=16,
                 window_size=16,
                 interval=3,
                 out_indices=-1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 output_cls_token=True,
                 sincos_pos_embed=False,
                 use_rel_pos_bias=False,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(ViTDet, self).__init__(init_cfg)

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'num_heads', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.num_layers = self.arch_settings['num_layers']
        self.num_heads = self.arch_settings['num_heads']
        self.interval = interval
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        # Set patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, window_size=window_size, in_chans=3, embed_dim=self.embed_dims)
        num_patches = self.patch_embed.num_patches

        self.grid_size = self.patch_embed.grid_size
        self.window_size = window_size

        if self.grid_size[0] % self.window_size:
            self.pad_l = 0
            self.pad_t = 0
            self.pad_r = (self.window_size - self.grid_size[1] % self.window_size) % self.window_size
            self.pad_b = (self.window_size - self.grid_size[0] % self.window_size) % self.window_size
        else:
            self.pad_l = 0
            self.pad_t = 0
            self.pad_r = 0
            self.pad_b = 0

        self.patch_size = patch_size
        self.out_indices = out_indices

        # Set cls token
        #self.output_cls_token = output_cls_token
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))

        # set sincos_pos_embed or pos_embed
        self.sincos_pos_embed = sincos_pos_embed
        self.interpolate_mode = interpolate_mode

        if self.sincos_pos_embed:
            self.build_2d_sincos_position_embedding()
        else:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches,
                            self.embed_dims))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
                assert out_indices[i] >= 0, f'Invalid out_indices {index}'
        self.out_indices = out_indices

        # stochastic depth decay rule
        dpr = np.linspace(0, drop_path_rate, self.arch_settings['num_layers'])

        self.blocks = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                dim=self.embed_dims,
                num_heads=self.num_heads,
                mlp_ratio=4.,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=0.,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=(self.window_size, self.window_size) if ((i + 1) % interval != 0) else tuple(self.grid_size),
                use_rel_pos_bias=use_rel_pos_bias)
            self.blocks.append(Block(**_layer_cfg))

        self.final_norm = final_norm
        if final_norm:
            self.norm = norm_layer(self.embed_dims)

        self._register_load_state_dict_pre_hook(self._prepare_checkpoint_hook)

    def init_weights(self):
        super(ViTDet, self).init_weights()

        if not (isinstance(self.init_cfg, dict)
                and self.init_cfg['type'] == 'Pretrained'):
            trunc_normal_(self.pos_embed, std=0.02)

    def _prepare_checkpoint_hook(self, state_dict, prefix, *args, **kwargs):
        name = prefix + 'pos_embed'
        if name not in state_dict.keys():
            return

        # sincos pos_embed
        if self.sincos_pos_embed:
            state_dict.pop(name)
            return
        else:
            # resize pos_embed
            ckpt_pos_embed_shape = state_dict[name].shape
            if self.pos_embed.shape != ckpt_pos_embed_shape:
                from mmcv.utils import print_log
                logger = get_root_logger()
                print_log(
                    f'Resize the pos_embed shape from {ckpt_pos_embed_shape} '
                    f'to {self.pos_embed.shape}.',
                    logger=logger)

                ckpt_pos_embed_shape = to_2tuple(
                    int(np.sqrt(ckpt_pos_embed_shape[1] - self.num_extra_tokens)))
                pos_embed_shape = self.grid_size

                state_dict[name] = self.resize_pos_embed(state_dict[name],
                                                    ckpt_pos_embed_shape,
                                                    pos_embed_shape,
                                                    self.interpolate_mode,
                                                    self.num_extra_tokens)


    @staticmethod
    def resize_pos_embed(pos_embed,
                         src_shape,
                         dst_shape,
                         mode='bicubic',
                         num_extra_tokens=1):
        """Resize pos_embed weights.

        Args:
            pos_embed (torch.Tensor): Position embedding weights with shape
                [1, L, C].
            src_shape (tuple): The resolution of downsampled origin training
                image.
            dst_shape (tuple): The resolution of downsampled new training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'bicubic'``
        Return:
            torch.Tensor: The resized pos_embed of shape [1, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
        _, L, C = pos_embed.shape
        src_h, src_w = src_shape
        assert L == src_h * src_w + num_extra_tokens
        extra_tokens = pos_embed[:, :num_extra_tokens]

        src_weight = pos_embed[:, num_extra_tokens:]
        src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

        dst_weight = F.interpolate(
            src_weight, size=dst_shape, align_corners=False, mode=mode)
        dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
		
        return dst_weight
        #return torch.cat((extra_tokens, dst_weight), dim=1)

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dims % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dims // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        #assert self.num_extra_tokens == 1, "Assuming one and only one token, [cls]"
        #pe_token = torch.zeros([1, 1, self.embed_dims], dtype=torch.float32)
        #self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed = nn.Parameter(pos_emb)
        self.pos_embed.requires_grad = False

    def window_partition(self, x, grid_size):
        B, L, C = x.shape
        H, W = grid_size[0], grid_size[1]

        x = x.reshape(B, H, W, C)
        if H % self.window_size:
            x = F.pad(x, (0, 0, self.pad_l, self.pad_r, self.pad_t, self.pad_b))
        _, Hp, Wp, _ = x.shape
        x = x.view(B, Hp // self.window_size, self.window_size, Wp // self.window_size, self.window_size, C)

        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        x = x.view(-1, self.window_size * self.window_size, C)
        return x

    def window_reverse(self, x, grid_size):
        B, L, C = x.shape
        H, W = grid_size[0], grid_size[1]
        
        x = x.view(-1, self.window_size, self.window_size, C)

        B = int(B / ((H + self.pad_t + self.pad_b) * (W + self.pad_l + self.pad_r) / self.window_size / self.window_size))
        x = x.view(B, (H + self.pad_t + self.pad_b) // self.window_size, (W + self.pad_l + self.pad_r) // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, (H + self.pad_t + self.pad_b), (W + self.pad_l + self.pad_r), -1)

        if H % self.window_size:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, -1)

        return x  

    def forward(self, x):
        x = self.patch_embed(x)

        #x = x + self.pos_embed[:, self.num_extra_tokens:]
        x = x + self.pos_embed

        x = self.drop_after_pos(x)

        # window_partition
        x = self.window_partition(x, self.grid_size)

        outs = []
        for i, layer in enumerate(self.blocks):
            # local self-attention & global self-attention
            if (i+1) % self.interval == 0:
                x  = self.window_reverse(x, self.grid_size)
                x = layer(x)
                x = self.window_partition(x, self.grid_size)
            else:
                x = layer(x)

            if i in self.out_indices:
                # window_reverse
                out = self.window_reverse(x, self.grid_size)

                if i == len(self.blocks) - 1:
                    if self.final_norm:
                        out = self.norm(out)
                        
                B, _, C = out.shape
                out = out.reshape(B, self.grid_size[0], self.grid_size[1], C).permute(0, 3, 1, 2).contiguous()

                outs.append(out)

        return tuple(outs)
