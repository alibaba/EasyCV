# Copyright (c) Alibaba, Inc. and its affiliates.
import csv
import json
import logging
import os
import traceback

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file.image import load_image

IGNORE_TAGS = ['*', '###']


@DATASOURCES.register_module()
class OCRDetSource(object):
    """ocr det data source
    """

    def __init__(self,
                 label_file,
                 data_dir='',
                 test_mode=False,
                 delimiter='\t'):
        """

        Args:
            label_file (str): path of label file
            data_dir (str, optional): folder of imgge data. Defaults to ''.
            test_mode (bool, optional): whether train or test. Defaults to False.
            delimiter (str, optional): delimiter used to separate elements in each row. Defaults to '\t'.
        """
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

    def label_encode(self, data):
        label = data['label']
        label = json.loads(label)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            if txt in IGNORE_TAGS:
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

    def parse(self, data_line):

        data_line = data_line.decode('utf-8')
        substr = data_line.strip('\n').split(self.delimiter)
        file_name = substr[0]
        label = substr[1]

        return file_name, label

    def __getitem__(self, idx):
        data_line = self.data_lines[idx]
        try:
            file_name, label = self.parse(data_line)
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception('{} does not exist!'.format(img_path))

            img = load_image(img_path, mode='BGR')
            data['img'] = img.astype(np.float32)
            data['ori_img_shape'] = img.shape
            outs = self.label_encode(data)
        except:
            logging.error(
                'When parsing line {}, error happened with msg: {}'.format(
                    data_line, traceback.format_exc()))
            outs = None
        if outs is None:
            rnd_idx = np.random.randint(
                len(self)) if not self.test_mode else (idx + 1) % len(self)
            return self[rnd_idx]
        return outs

    def __len__(self):
        return len(self.data_lines)


@DATASOURCES.register_module()
class OCRPaiDetSource(OCRDetSource):
    """ocr det data source for pai format
    """

    def __init__(self, label_file, data_dir='', test_mode=False):
        """

        Args:
            label_file (str or list[str]): path of label file
            data_dir (str, optional): folder of imgge data. Defaults to ''.
            test_mode (bool, optional): whether train or test. Defaults to False.
        """
        super(OCRPaiDetSource, self).__init__(
            label_file, data_dir=data_dir, test_mode=test_mode)

    def get_image_info_list(self, label_file):
        data_lines = []
        if type(label_file) == list:
            for file in label_file:
                data_lines += list(csv.reader(open(file)))[1:]
        else:
            data_lines = list(csv.reader(open(label_file)))[1:]
        return data_lines

    def label_encode(self, data):
        label = data['label']
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(nBox):
            box = label[bno]['coord']
            box = [int(float(pos)) for pos in box]
            box = [box[idx:idx + 2] for idx in range(0, 8, 2)]
            txt = json.loads(label[bno]['text'])['text']
            boxes.append(box)
            txts.append(txt)
            if txt in IGNORE_TAGS:
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

    def parse(self, data_line):

        file_name = json.loads(data_line[1])['tfspath'].split('/')[-1]
        label = json.loads(data_line[2])[0]

        return file_name, label
