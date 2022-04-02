# Copyright (c) Alibaba, Inc. and its affiliates.
from functools import partial

import torch
import torch.nn as nn
from packaging import version
from timm.models.vision_transformer import Block

from easycv.models.utils import get_2d_sincos_pos_embed
from ..registry import NECKS
from ..utils import _init_weights, build_norm_layer, trunc_normal_


@NECKS.register_module
class DINONeck(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 use_bn=False,
                 norm_last_layer=True,
                 nlayers=3,
                 hidden_dim=2048,
                 bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
                # layers.append(build_norm_layer(dict(type='SyncBN'), hidden_dim)[1])
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                    # layers.append(build_norm_layer(dict(type='SyncBN'), hidden_dim)[1])

                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


@NECKS.register_module
class MoBYMLP(nn.Module):

    def __init__(self,
                 in_channels=256,
                 hid_channels=4096,
                 out_channels=256,
                 num_layers=2,
                 with_avg_pool=True):
        super(MoBYMLP, self).__init__()

        # hidden layers
        linear_hidden = [nn.Identity()]
        for i in range(num_layers - 1):
            linear_hidden.append(
                nn.Linear(in_channels if i == 0 else hid_channels,
                          hid_channels))
            linear_hidden.append(nn.BatchNorm1d(hid_channels))
            linear_hidden.append(nn.ReLU(inplace=True))
        self.linear_hidden = nn.Sequential(*linear_hidden)
        self.linear_out = nn.Linear(
            in_channels if num_layers == 1 else hid_channels,
            out_channels) if num_layers >= 1 else nn.Identity()
        self.with_avg_pool = True
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = x[0]
        if self.with_avg_pool and len(x.shape) == 4:
            bs = x.shape[0]
            x = self.avg_pool(x).view([bs, -1])
        # print(x.shape)
        # exit()
        x = self.linear_hidden(x)
        x = self.linear_out(x)
        return [x]

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)


@NECKS.register_module
class NonLinearNeckSwav(nn.Module):
    '''The non-linear neck in byol: fc-syncbn-relu-fc
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 export=False):
        super(NonLinearNeckSwav, self).__init__()

        if version.parse(torch.__version__) < version.parse('1.4.0'):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.export = export
        if not self.export:
            _, self.bn0 = build_norm_layer(dict(type='SyncBN'), hid_channels)
        else:
            _, self.bn0 = build_norm_layer(dict(type='BN'), hid_channels)

        self.fc0 = nn.Linear(in_channels, hid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(hid_channels, out_channels)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        # syncbn < torch1.4.0 or bn while export need unsqueeze 4D dims
        if self.expand_for_syncbn or self.export:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2, 'Got: {}'.format(
            len(x))  # fit for vit model
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self._forward_syncbn(self.bn0, x)
        x = self.relu(x)
        x = self.fc1(x)

        return [x]


@NECKS.register_module
class NonLinearNeckV0(nn.Module):
    '''The non-linear neck in ODC, fc-bn-relu-dropout-fc-relu
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(NonLinearNeckV0, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse('1.4.0'):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc0 = nn.Linear(in_channels, hid_channels)
        if sync_bn:
            _, self.bn0 = build_norm_layer(
                dict(type='SyncBN', momentum=0.001, affine=False),
                hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(
                hid_channels, momentum=0.001, affine=False)

        self.fc1 = nn.Linear(hid_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2  # to fit vit model
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.relu(x)
        return [x]


@NECKS.register_module
class NonLinearNeckV1(nn.Module):
    '''The non-linear neck in MoCO v2: fc-relu-fc
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        # assert len(x) == 1 or len(x)==2  # to fit vit model, vit model extract 2 features, we use first
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckV2(nn.Module):
    '''The non-linear neck in byol: fc-bn-relu-fc
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2, 'Got: {}'.format(
            len(x))  # to fit vit model
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


@NECKS.register_module
class NonLinearNeckSimCLR(nn.Module):
    '''SimCLR non-linear neck.

    Structure: fc(no_bias)-bn(has_bias)-[relu-fc(no_bias)-bn(no_bias)].
        The substructures in [] can be repeated. For the SimCLR default setting,
        the repeat time is 1.

    However, PyTorch does not support to specify (weight=True, bias=False).
        It only support \"affine\" including the weight and bias. Hence, the
        second BatchNorm has bias in this implementation. This is different from
        the offical implementation of SimCLR.

    Since SyncBatchNorm in pytorch<1.4.0 does not support 2D input, the input is
        expanded to 4D with shape: (N,C,1,1). I am not sure if this workaround
        has no bugs. See the pull request here:
        https://github.com/pytorch/pytorch/pull/29626

    Args:
        in_channels: input channel number
        hid_channels: hidden channels
        out_channels: output channel number
        num_layers (int): number of fc layers, it is 2 in the SimCLR default setting.
        with_avg_pool:  output with average pooling
    '''

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_layers=2,
                 with_avg_pool=True):
        super(NonLinearNeckSimCLR, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse('1.4.0'):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=False)
        _, self.bn0 = build_norm_layer(dict(type='SyncBN'), hid_channels)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 \
                else hid_channels
            self.add_module('fc{}'.format(i),
                            nn.Linear(hid_channels, this_channels, bias=False))
            self.add_module(
                'bn{}'.format(i),
                build_norm_layer(dict(type='SyncBN'), this_channels)[1])
            self.fc_names.append('fc{}'.format(i))
            self.bn_names.append('bn{}'.format(i))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2  # to fit vit model
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        x = self._forward_syncbn(self.bn0, x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            bn = getattr(self, bn_name)
            x = self.relu(x)
            x = fc(x)
            x = self._forward_syncbn(bn, x)
        return [x]


@NECKS.register_module
class RelativeLocNeck(nn.Module):
    '''Relative patch location neck: fc-bn-relu-dropout
    '''

    def __init__(self,
                 in_channels,
                 out_channels,
                 sync_bn=False,
                 with_avg_pool=True):
        super(RelativeLocNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if version.parse(torch.__version__) < version.parse('1.4.0'):
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.fc = nn.Linear(in_channels * 2, out_channels)
        if sync_bn:
            _, self.bn = build_norm_layer(
                dict(type='SyncBN', momentum=0.003), out_channels)
        else:
            self.bn = nn.BatchNorm1d(out_channels, momentum=0.003)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear, std=0.005, bias=0.1)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1 or len(x) == 2  # to fit vit model
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn, x)
        else:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return [x]


@NECKS.register_module
class MAENeck(nn.Module):
    """MAE decoder

    Args:
        num_patches(int): number of patches from encoder
        embed_dim(int): encoder embedding dimension
        patch_size(int): encoder patch size
        in_chans(int): input image channels
        decoder_embed_dim(int): decoder embedding dimension
        decoder_depth(int): number of decoder layers
        decoder_num_heads(int): Parallel attention heads
        mlp_ratio(float): mlp ratio
        norm_layer: type of normalization layer
    """

    def __init__(self,
                 num_patches,
                 embed_dim=768,
                 patch_size=16,
                 in_chans=3,
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 mlp_ratio=4.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1],
            int(num_patches**.5),
            cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        self.decoder_blocks = nn.ModuleList([
            Block(
                decoder_embed_dim,
                decoder_num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=norm_layer) for i in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True)

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

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(
            x_,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x
