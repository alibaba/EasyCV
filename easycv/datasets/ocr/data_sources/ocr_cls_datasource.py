# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import traceback

import numpy as np

from easycv.datasets.ocr.data_sources.ocr_det_datasource import OCRDetSource
from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image


@DATASOURCES.register_module(force=True)
class OCRClsSource(OCRDetSource):
    """ocr direction classification data source
    """

    def __init__(self,
                 label_file,
                 data_dir='',
                 test_mode=False,
                 delimiter='\t',
                 label_list=['0', '180']):
        super(OCRClsSource, self).__init__(
            label_file,
            data_dir=data_dir,
            test_mode=test_mode,
            delimiter=delimiter)
        self.label_list = label_list

    def label_encode(self, data):
        label = data['label']
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data['label'] = label
        return data
