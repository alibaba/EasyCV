# Copyright (c) Alibaba, Inc. and its affiliates.
from .coco import PoseTopDownSourceCoco
from .hand import HandCocoPoseTopDownSource
from .top_down import PoseTopDownSource
from .wholebody import WholeBodyCocoTopDownSource

__all__ = [
    'PoseTopDownSourceCoco', 'PoseTopDownSource', 'HandCocoPoseTopDownSource',
    'WholeBodyCocoTopDownSource'
]
