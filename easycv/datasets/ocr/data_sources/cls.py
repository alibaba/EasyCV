# Copyright (c) Alibaba, Inc. and its affiliates.
# debug
import sys
sys.path.append('/root/code/ocr/EasyCV')

import os
import logging
import numpy as np
import traceback

from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image
from easycv.datasets.ocr.data_sources.det import OCRDetSource


@DATASOURCES.register_module(force=True)
class OCRClsSource(OCRDetSource):
    """ocr direction classification data source
    """

    def __init__(self, label_file, data_dir="", test_mode=False, delimiter='\t', label_list=['0','180']):
        super(OCRClsSource, self).__init__(
            label_file, data_dir=data_dir, test_mode=test_mode, delimiter=delimiter)
        self.label_list = label_list
        
    def label_encode(self, data):
        label = data['label']
        if label not in self.label_list:
            return None
        label = self.label_list.index(label)
        data['label'] = label
        return data
    
if __name__=="__main__":
    data_source = OCRClsSource(label_file='/nas/database/ocr/direction/pai/label_file/test_direction.txt',data_dir='/nas/database/ocr/direction/pai/img/test')
    res =data_source.get_sample(2)
    print(res)