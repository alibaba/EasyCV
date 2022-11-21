# Copyright (c) Alibaba, Inc. and its affiliates.
from .raw import SegSourceRaw
from .voc import SegSourceVoc2007, SegSourceVoc2010, SegSourceVoc2012

__all__ = [
    'SegSourceRaw', 'SegSourceVoc2010', 'SegSourceVoc2007', 'SegSourceVoc2012'
]
