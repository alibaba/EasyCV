# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import traceback

import numpy as np

from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module(force=True)
class OCRDetDataset(BaseDataset):
    """Dataset for ocr text detection
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRDetDataset, self).__init__(
            data_source, pipeline, profiling=profiling)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, idx):
        try:
            data_dict = self.data_source[idx]
            data_dict = self.pipeline(data_dict)
        except:
            logging.error(
                'When parsing line {}, error happened with msg: {}'.format(
                    idx, traceback.format_exc()))
            data_dict = None
        if data_dict is None:
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
        return data_dict

    def evaluate(self, results, evaluators, logger=None, **kwargs):
        assert len(evaluators) == 1, \
            'ocrdet evaluation only support one evaluator'
        points = results.pop('points')
        ignore_tags = results.pop('ignore_tags')
        polys = results.pop('polys')
        eval_res = evaluators[0].evaluate(points, polys, ignore_tags)

        return eval_res
