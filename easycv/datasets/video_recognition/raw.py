# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import traceback

import numpy as np

from easycv.datasets.registry import DATASETS
from easycv.datasets.shared.base import BaseDataset


@DATASETS.register_module
class VideoDataset(BaseDataset):
    """Dataset for video recognition
    """

    def __init__(self, data_source, pipeline, profiling=False):
        """

        Args:
            data_source: Data_source config dict
            pipeline: Pipeline config list
            profiling: If set True, will print pipeline time
        """
        super(VideoDataset, self).__init__(
            data_source, pipeline, profiling=profiling)

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
            return self[np.random.randint(len(self))]
        return data_dict

    def evaluate(self, results, evaluators=[], **kwargs):
        """Evaluate the dataset.
        """
        assert len(evaluators) == 1, \
            'classification evaluation only support one evaluator'
        gt_labels = results.pop('label')
        eval_res = evaluators[0].evaluate(results, gt_labels)

        return eval_res
