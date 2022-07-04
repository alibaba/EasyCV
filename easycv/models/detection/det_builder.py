# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.models.builder import build
from easycv.models.registry import Registry

DETRTRANSFORMER = Registry('detr_transformer')


def build_detr_transformer(cfg):
    return build(cfg, DETRTRANSFORMER)
