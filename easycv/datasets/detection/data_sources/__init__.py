# Copyright (c) Alibaba, Inc. and its affiliates.
from .coco import DetSourceCoco, DetSourceCoco2017
from .coco_panoptic import DetSourceCocoPanoptic
from .pai_format import DetSourcePAI
from .raw import DetSourceRaw
from .voc import DetSourceVOC, DetSourceVOC2007, DetSourceVOC2012

__all__ = [
    'DetSourceCoco', 'DetSourceCocoPanoptic', 'DetSourcePAI', 'DetSourceRaw',
    'DetSourceVOC', 'DetSourceVOC2007', 'DetSourceVOC2012', 'DetSourceCoco2017'
]
