# Copyright (c) Alibaba, Inc. and its affiliates.
from .coco import SegSourceCoco, SegSourceCoco2017
from .raw import SegSourceRaw
from .voc import SegSourceVoc2007, SegSourceVoc2010, SegSourceVoc2012

__all__ = [
    'SegSourceRaw', 'SegSourceVoc2010', 'SegSourceVoc2007', 'SegSourceVoc2012',
    'SegSourceCoco', 'SegSourceCoco2017'
]
