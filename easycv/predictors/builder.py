# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.utils.registry import Registry, build_from_cfg

PREDICTORS = Registry('predictor')


def build_predictor(cfg, default_args=None):
    return build_from_cfg(cfg, PREDICTORS, default_args=default_args)
