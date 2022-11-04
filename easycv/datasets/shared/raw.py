# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.registry import DATASETS
from easycv.framework.errors import NotImplementedError
from .base import BaseDataset


@DATASETS.register_module
class RawDataset(BaseDataset):

    def __init__(self, data_source, pipeline, profiling=False):
        super(RawDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        results = self.data_source[idx]
        return self.pipeline(results)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplementedError
