# Copyright (c) Alibaba, Inc. and its affiliates.
from easycv.datasets.registry import DATASETS
from .base import BaseDataset


@DATASETS.register_module
class RawDataset(BaseDataset):
    # TODO: remove `with_label`, return result_dict for self.data_source.get_sample(idx), pipeline receives dict
    # like: results = self.pipeline(self.data_source.get_sample(idx))
    def __init__(self, data_source, pipeline, with_label=False):
        self.with_label = with_label
        super(RawDataset, self).__init__(data_source, pipeline)

    def __getitem__(self, idx):
        # TODO
        if self.with_label:
            img, label = self.data_source.get_sample(idx)
            img = self.pipeline(img)
            return dict(img=img, gt_label=label)
        else:
            img = self.data_source.get_sample(idx)
            img = self.pipeline(img)
            return dict(img=img)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplementedError
