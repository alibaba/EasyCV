# Copyright (c) Alibaba, Inc. and its affiliates.
from .cityscapes import SegSourceCityscapes
from .coco import SegSourceCoco, SegSourceCoco2017
from .coco_stuff import SegSourceCocoStuff10k, SegSourceCocoStuff164k
from .raw import SegSourceRaw
from .voc import SegSourceVoc2007, SegSourceVoc2010, SegSourceVoc2012

__all__ = [
    'SegSourceRaw', 'SegSourceVoc2010', 'SegSourceVoc2007', 'SegSourceVoc2012',
    'SegSourceCoco', 'SegSourceCoco2017', 'SegSourceCocoStuff164k',
    'SegSourceCocoStuff10k', 'SegSourceCityscapes'
]
