# Copyright (c) Alibaba, Inc. and its affiliates.
from torch import nn

from easycv.utils.registry import build_from_cfg
from .registry import (ATTENTION, BACKBONES, FEEDFORWARD_NETWORK,
                       FUSION_LAYERS, HEADS, LOSSES, MIDDLE_ENCODERS, MODELS,
                       NECKS, POSITIONAL_ENCODING, TRANSFORMER,
                       TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE,
                       VOXEL_ENCODERS)


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_model(cfg):
    return build(cfg, MODELS)


def build_voxel_encoder(cfg):
    """Build voxel encoder."""
    return VOXEL_ENCODERS.build(cfg)


def build_middle_encoder(cfg):
    """Build middle level encoder."""
    return MIDDLE_ENCODERS.build(cfg)


def build_fusion_layer(cfg):
    """Build fusion layer."""
    return FUSION_LAYERS.build(cfg)


def build_transformer(cfg, default_args=None):
    """Builder for Transformer."""
    return build_from_cfg(cfg, TRANSFORMER, default_args)


def build_positional_encoding(cfg, default_args=None):
    """Builder for Position Encoding."""
    return build_from_cfg(cfg, POSITIONAL_ENCODING, default_args)


def build_attention(cfg, default_args=None):
    """Builder for attention."""
    return build_from_cfg(cfg, ATTENTION, default_args)


def build_feedforward_network(cfg, default_args=None):
    """Builder for feed-forward network (FFN)."""
    return build_from_cfg(cfg, FEEDFORWARD_NETWORK, default_args)


def build_transformer_layer(cfg, default_args=None):
    """Builder for transformer layer."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER, default_args)


def build_transformer_layer_sequence(cfg, default_args=None):
    """Builder for transformer encoder and transformer decoder."""
    return build_from_cfg(cfg, TRANSFORMER_LAYER_SEQUENCE, default_args)
