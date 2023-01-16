# Copyright (c) OpenMMLab. All rights reserved.

import json
import os
from glob import glob
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .base import DetSourceBase


def parse_json(source_item, classes):
    img_path, target_path = source_item
    with io.open(target_path, 'r') as t:
        info = json.load(t)
        img_name = info.get('asset')['name']
        gt_bboxes = []
        gt_labels = []
        for obj in info.get('regions'):
            cls_id = classes.index(obj['tags'][0])
            bbox = obj['boundingBox']

            box = [
                float(bbox['left']),
                float(bbox['top']),
                float(bbox['left'] + bbox['width']),
                float(bbox['top'] + bbox['height'])
            ]
            gt_bboxes.append(box)
            gt_labels.append(cls_id)

    if len(gt_bboxes) == 0:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)

    img_info = {
        'gt_bboxes': np.array(gt_bboxes, dtype=np.float32),
        'gt_labels': np.array(gt_labels, dtype=np.int64),
        'filename':
        os.path.dirname(target_path).replace('annotations', img_name)
    }

    return img_info


@DATASOURCES.register_module
class DetSourceArtaxor(DetSourceBase):
    """
    data dir is as follows:
    ```
    |- data
        |-Images
            |-000040.jpg
            |-...
        |-Annotations
            |-000040.jpg.txt
            |-...
        |-train.txt
        |-val.txt
        |-...

    ```
    Example1:
        data_source = DetSourceWiderPerson(
            path='/your/data/train.txt',
            classes=${VOC_CLASSES},
        )
    Example1:
        data_source = DetSourceWiderPerson(
            path='/your/voc_data/train.txt',
            classes=${CLASSES},
            img_root_path='/your/data/Images',
            img_root_path='/your/data/Annotations'
        )
    """
    CLASSES = [
        'Araneae', 'Coleoptera', 'Diptera', 'Hemiptera', 'Hymenoptera',
        'Lepidoptera', 'Odonata'
    ]

    def __init__(self,
                 path,
                 classes=CLASSES,
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 label_suffix='.json',
                 parse_fn=parse_json,
                 num_processes=int(cpu_count() / 2),
                 **kwargs) -> None:
        """
        Args:
            path: path of img id list file in root
            classes: classes list
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            label_suffix: suffix of label file
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.path = path
        self.label_suffix = label_suffix

        super(DetSourceArtaxor, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):

        assert os.path.exists(self.path), f'{self.path} is not exists'

        imgs_path_list = []
        labels_path_list = []

        for category in self.CLASSES:
            img_path = os.path.join(self.path, category)
            assert os.path.exists(img_path), f'{img_path} is not exists'
            label_list = glob(
                os.path.join(img_path, 'annotations', '*' + self.label_suffix))
            for label_path in label_list:

                imgs_path_list.append(category)
                labels_path_list.append(label_path)

        return list(zip(imgs_path_list, labels_path_list))
