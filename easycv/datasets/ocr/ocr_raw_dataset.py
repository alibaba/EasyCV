# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import traceback

import numpy as np

from easycv.datasets.shared.base import BaseDataset


class OCRRawDataset(BaseDataset):
    """Dataset for ocr
    """

    def __init__(self, data_source, pipeline, profiling=False):
        super(OCRRawDataset, self).__init__(
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
        pass
