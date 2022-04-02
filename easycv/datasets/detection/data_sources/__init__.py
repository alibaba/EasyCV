# Copyright (c) Alibaba, Inc. and its affiliates.
from .coco import DetSourceCoco
from .pai_format import DetSourcePAI
from .raw import DetSourceRaw
from .voc import DetSourceVOC

__all__ = ['DetSourceCoco', 'DetSourcePAI', 'DetSourceRaw', 'DetSourceVOC']
