# Copyright (c) Alibaba, Inc. and its affiliates.
from torch import nn

# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.utils.registry import Registry, build_from_cfg

DETRTRANSFORMER = Registry('detr_transformer')


def build(cfg, registry, default_args=None):
    if isinstance(cfg, list):
        modules = [
            build_from_cfg(cfg_, registry, default_args) for cfg_ in cfg
        ]
        return nn.Sequential(*modules)
    else:
        return build_from_cfg(cfg, registry, default_args)


def build_detr_transformer(cfg):
    return build(cfg, DETRTRANSFORMER)
