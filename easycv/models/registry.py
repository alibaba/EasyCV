# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.utils.registry import Registry

MODELS = Registry('model')
BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
VOXEL_ENCODERS = Registry('voxel_encoder')
MIDDLE_ENCODERS = Registry('middle_encoder')
FUSION_LAYERS = Registry('fusion_layers')
TRANSFORMER = Registry('Transformer')
TRANSFORMER_LAYER = Registry('transformerLayer')
TRANSFORMER_LAYER_SEQUENCE = Registry('transformer-layers sequence')
POSITIONAL_ENCODING = Registry('position encoding')
ATTENTION = Registry('attention')
FEEDFORWARD_NETWORK = Registry('feed-forward Network')
