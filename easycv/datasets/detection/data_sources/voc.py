# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.detection.data_sources.base import DetSourceBase
from easycv.datasets.registry import DATASOURCES
from easycv.file import io

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']


def parse_xml(source_item, classes):
    img_path, xml_path = source_item
    with io.open(xml_path, 'r') as f:
        tree = ET.parse(f)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)
        gt_bboxes = []
        gt_labels = []
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if int(difficult) == 1:
                continue
            if cls not in classes:
                logging.warning(
                    'class: %s not in given class list, skip the object!' %
                    cls)
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            box = (float(xmlbox.find('xmin').text),
                   float(xmlbox.find('ymin').text),
                   float(xmlbox.find('xmax').text),
                   float(xmlbox.find('ymax').text))
            gt_bboxes.append(box)
            gt_labels.append(cls_id)

    if len(gt_bboxes) == 0:
        gt_bboxes = np.zeros((0, 5), dtype=np.float32)

    img_info = {
        'gt_bboxes': np.array(gt_bboxes, dtype=np.float32),
        'gt_labels': np.array(gt_labels, dtype=np.int64),
        'filename': img_path,
    }

    return img_info


@DATASOURCES.register_module
class DetSourceVOC(DetSourceBase):
    """
    data dir is as follows:
    ```
    |- voc_data
        |-ImageSets
            |-Main
                |-train.txt
                |-...
        |-JPEGImages
            |-00001.jpg
            |-...
        |-Annotations
            |-00001.xml
            |-...

    ```
    Example1:
        data_source = DetSourceVOC(
            path='/your/voc_data/ImageSets/Main/train.txt',
            classes=${VOC_CLASSES},
        )
    Example1:
        data_source = DetSourceVOC(
            path='/your/voc_data/train.txt',
            classes=${VOC_CLASSES},
            img_root_path='/your/voc_data/images',
            img_root_path='/your/voc_data/annotations'
        )
    """

    def __init__(self,
                 path,
                 classes=[],
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
            path: path of img id list file in ImageSets/Main/
            classes: classes list
            img_root_path: image dir path, if None, default to detect the image dir by the relative path of the `path`
                according to the VOC data format.
            label_root_path: label dir path, if None, default to detect the label dir by the relative path of the `path`
                according to the VOC data format.
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            img_suffix: suffix of image file
            label_suffix: suffix of label file
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.path = path
        self.img_root_path = img_root_path
        self.label_root_path = label_root_path
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        super(DetSourceVOC, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):
        if not self.img_root_path:
            self.img_root_path = os.path.join(
                self.path.split('ImageSets/Main')[0], 'JPEGImages')
        if not self.label_root_path:
            self.label_root_path = os.path.join(
                self.path.split('ImageSets/Main')[0], 'Annotations')

        imgs_path_list = []
        labels_path_list = []
        with io.open(self.path, 'r') as t:
            id_lines = t.read().splitlines()
            for id_line in id_lines:
                img_id = id_line.strip().split(' ')[0]
                img_path = os.path.join(self.img_root_path,
                                        img_id + self.img_suffix)
                imgs_path_list.append(img_path)
                label_path = os.path.join(self.label_root_path,
                                          img_id + self.label_suffix)
                labels_path_list.append(label_path)

        return list(zip(imgs_path_list, labels_path_list))
