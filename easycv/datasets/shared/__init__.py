# Copyright (c) Alibaba, Inc. and its affiliates.
import torch

from ..shared.multi_view import MultiViewDataset
from . import data_sources  # pylint: disable=unused-import
from . import pipelines  # pylint: disable=unused-import
from .base import BaseDataset
from .dataset_wrappers import ConcatDataset, RepeatDataset
from .odps_reader import OdpsReader
from .raw import RawDataset

__all__ = [
    'ConcatDataset', 'RepeatDataset', 'OdpsReader', 'RawDataset',
    'BaseDataset', 'MultiViewDataset'
]

# TODO: merge `DaliImageNetTFRecordDataSet` and `DaliTFRecordMultiViewDataset`
# avoid to import dali on cpu env which will result error
if torch.cuda.is_available():
    from .dali_tfrecord_imagenet import DaliImageNetTFRecordDataSet
    from .dali_tfrecord_multi_view import DaliTFRecordMultiViewDataset

    __all__.extend(
        ['DaliImageNetTFRecordDataSet', 'DaliTFRecordMultiViewDataset'])
