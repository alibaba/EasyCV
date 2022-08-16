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


@DATASOURCES.register_module(force=True)
class OCRRECSource(object):
    """ocr rec data source
    """
    
    def __init__(self, label_file, data_dir="", ext_data_num=0, test_mode=False, delimiter='\t'):
        self.data_dir = data_dir
        self.delimiter = delimiter
        self.test_mode = test_mode
        self.ext_data_num = ext_data_num
        self.data_lines = self.get_image_info_list(label_file)
        
    def get_image_info_list(self, label_file):
        data_lines = []
        with open(label_file, 'rb') as f:
            lines = f.readlines()
            data_lines.extend(lines)
        return data_lines
    
    def get_sample(self, idx, get_ext=True):
        data_line = self.data_lines[idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            
            outs = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception('{} does not exist!'.format(img_path))
            img = load_image(img_path, mode='BGR')
            outs['img'] = img.astype(np.float32)
            outs['ori_img_shape'] = img.shape
            if get_ext:
                outs['ext_data'] = self.get_ext_data()
            return outs
        except:
            logging.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            rnd_idx = np.random.randint(self.__len__(
            )) if not self.test_mode else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        
    def __len__(self):
        return len(self.data_lines)
    
    def get_ext_data(self):
        ext_data = []
        
        while len(ext_data) < self.ext_data_num:
            data = self.get_sample(np.random.randint(self.__len__()),get_ext=False)
            ext_data.append(data)
        return ext_data
        
        
            
        
    
if __name__=="__main__":
    data_source = OCRRECSource(label_file='/mnt/data/database/ocr/rec/ic15_data/rec_gt_test.txt',data_dir='/mnt/data/database/ocr/rec/ic15_data',ext_data_num=2)
    result = data_source.get_sample(10)
    print(result.keys())
    print(result['ext_data'])