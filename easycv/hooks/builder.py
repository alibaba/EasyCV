# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.utils.registry import build_from_cfg
from .registry import HOOKS


def build_hook(cfg, default_args=None):
    return build_from_cfg(cfg, HOOKS, default_args)
