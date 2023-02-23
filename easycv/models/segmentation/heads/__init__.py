# Copyright (c) Alibaba, Inc. and its affiliates.
from .fcn_head import FCNHead
from .mask2former_head import Mask2FormerHead
from .segformer_head import SegformerHead
from .stdc_head import STDCHead
from .uper_head import UPerHead

__all__ = [
    'FCNHead', 'UPerHead', 'Mask2FormerHead', 'SegformerHead', 'STDCHead'
]
