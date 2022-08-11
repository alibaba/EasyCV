# Copyright (c) Alibaba, Inc. and its affiliates.
# debug
import sys
sys.path.append('/root/code/ocr/EasyCV')

import json
import os
import numpy as np
import logging
import traceback

from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image


@DATASOURCES.register_module()
class OCRDetSource(object):
    """ocr det data source
    """
    
    def __init__(self, label_file, data_dir="", test_mode=False, delimiter='\t'):
        self.data_dir = data_dir
        self.delimiter = delimiter
        self.test_mode = test_mode
        self.data_lines = self.get_image_info_list(label_file)
        
        
    def get_image_info_list(self, label_file):
        data_lines = []
        with open(label_file, 'rb') as f:
            lines = f.readlines()
            data_lines.extend(lines)
        return data_lines
    
    def detlabel_encode(self, data):
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in ['*', '###']:
                txt_tags.append(True)
            else:
                txt_tags.append(False)
        if len(boxes) == 0:
            return None
        boxes = self.expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool)

        data['polys'] = boxes
        data['texts'] = txts
        data['ignore_tags'] = txt_tags
        return data
        
    def expand_points_num(self, boxes):
        max_points_num = 0
        for box in boxes:
            if len(box) > max_points_num:
                max_points_num = len(box)
        ex_boxes = []
        for box in boxes:
            ex_box = box + [box[-1]] * (max_points_num - len(box))
            ex_boxes.append(ex_box)
        return ex_boxes
    
    def get_sample(self, idx):
        data_line = self.data_lines[idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            
            img = load_image(img_path, mode='BGR')
            data['img'] = img.astype(np.float32)
            outs = self.detlabel_encode(data)
        except:
            logging.error(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            rnd_idx = np.random.randint(self.__len__(
            )) if not self.test_mode else (idx + 1) % self.__len__()
            return self.__getitem__(rnd_idx)     
        return outs
    
    def __len__(self):
        return len(self.data_lines)
    

if __name__=="__main__":
    data_source = OCRDetSource(label_file='/root/code/ocr/EasyCV/train_data/icdar2015/text_localization/test_icdar2015_label.txt',data_dir='/root/code/ocr/EasyCV/train_data/icdar2015/text_localization')
    print(len(data_source))
        
    
    