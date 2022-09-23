# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import traceback

import numpy as np

from easycv.datasets.registry import DATASETS
from .ocr_raw_dataset import OCRRawDataset


@DATASETS.register_module()
class OCRDetDataset(OCRRawDataset):
    """Dataset for ocr text detection
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRDetDataset, self).__init__(
            data_source, pipeline, profiling=profiling)

    def evaluate(self, results, evaluators, logger=None, **kwargs):
        assert len(evaluators) == 1, \
            'ocrdet evaluation only support one evaluator'
        points = results.pop('points')
        ignore_tags = results.pop('ignore_tags')
        polys = results.pop('polys')
        eval_res = evaluators[0].evaluate(points, polys, ignore_tags)

        return eval_res
