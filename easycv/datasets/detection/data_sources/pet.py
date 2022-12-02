# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .base import DetSourceBase


def parse_xml(source_item, classes):
    img_path, xml_path = source_item
    with io.open(xml_path[0], 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        gt_bboxes = []
        gt_labels = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            if int(difficult) == 1:
                continue
            cls_id = classes.index(int(xml_path[1]))
            xmlbox = obj.find('bndbox')
            box = (float(xmlbox.find('xmin').text),
                   float(xmlbox.find('ymin').text),
                   float(xmlbox.find('xmax').text),
                   float(xmlbox.find('ymax').text))
            gt_bboxes.append(box)
            gt_labels.append(cls_id)

    if len(gt_bboxes) == 0:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)

    img_info = {
        'gt_bboxes': np.array(gt_bboxes, dtype=np.float32),
        'gt_labels': np.array(gt_labels, dtype=np.int64),
        'filename': img_path,
    }

    return img_info


@DATASOURCES.register_module
class DetSourcePet(DetSourceBase):
    """
    data dir is as follows:
    ```
    |- data
        |-annotations
            |-annotations
                |-list.txt
                |-test.txt
                |-trainval.txt
                |-xmls
                    |-Abyssinian_6.xml
                    |-...
        |-images
            |-images
                |-Abyssinian_6.jpg
                |-...

    ```
    Example0:
        data_source = DetSourcePet(
            path='/your/data/annotations/annotations/trainval.txt',
            classes_id=1 or 2 or 3,
    Example1:
        data_source = DetSourcePet(
            path='/your/data/annotations/annotations/trainval.txt',
            classes_id=1 or 2 or 3,
            img_root_path='/your/data//images',
            img_root_path='/your/data/annotations/annotations/xmls'
        )
    """
    CLASSES_CFG = {
        #  1:37 Class ids
        1: list(range(1, 38)),
        # 1:Cat 2:Dog
        2: list(range(1, 3)),
        #  1-25:Cat 1:12:Dog
        3: list(range(1, 26))
    }

    def __init__(self,
                 path,
                 classes_id=1,
                 img_root_path=None,
                 label_root_path=None,
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 img_suffix='.jpg',
                 label_suffix='.xml',
                 parse_fn=parse_xml,
                 num_processes=int(cpu_count() / 2),
                 **kwargs):
        """
        Args:
            path: path of img id list file in pet format
            classes_id: 1= 1:37 Class ids, 2 = 1:Cat 2:Dog, 3 =  1-25:Cat 1:12:Dog
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            img_suffix: suffix of image file
            label_suffix: suffix of label file
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.classes_id = classes_id
        self.img_root_path = img_root_path
        self.label_root_path = label_root_path
        self.path = path
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix

        super(DetSourcePet, self).__init__(
            classes=self.CLASSES_CFG[classes_id],
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):
        if not self.img_root_path:
            self.img_root_path = os.path.join(self.path, '../../..',
                                              'images/images')
        if not self.label_root_path:
            self.label_root_path = os.path.join(
                os.path.dirname(self.path), 'xmls')

        assert os.path.exists(self.path), f'{self.path} is not exists'
        imgs_path_list = []
        labels_path_list = []

        with io.open(self.path, 'r') as t:
            id_lines = t.read().splitlines()
            for id_line in id_lines:
                img_id = id_line.strip()
                if img_id == '':
                    continue

                line = img_id.split()

                img_path = os.path.join(self.img_root_path,
                                        line[0] + self.img_suffix)

                label_path = os.path.join(self.label_root_path,
                                          line[0] + self.label_suffix)

                if not os.path.exists(label_path):
                    continue

                labels_path_list.append((label_path, line[self.classes_id]))
                imgs_path_list.append(img_path)

        return list(zip(imgs_path_list, labels_path_list))
