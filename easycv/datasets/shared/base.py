# Copyright (c) Alibaba, Inc. and its affiliates.
from abc import ABCMeta, abstractmethod

from torch.utils.data import Dataset

from easycv.utils import build_from_cfg
from ..builder import build_datasource
from ..registry import PIPELINES
from .pipelines.transforms import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base Dataset
    """

    def __init__(self, data_source, pipeline, profiling=False):
        self.data_source = build_datasource(data_source)
        pipeline = [build_from_cfg(p, PIPELINES) for p in pipeline]
        self.pipeline = Compose(pipeline, profiling=profiling)

    def __len__(self):
        return self.data_source.get_length()

    @abstractmethod
    def __getitem__(self, idx):
        pass

    @abstractmethod
    def evaluate(self, results, evaluators, logger=None, **kwargs):
        pass
