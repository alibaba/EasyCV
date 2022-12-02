# Copyright (c) OpenMMLab. All rights reserved.

import json
import os
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .base import DetSourceBase


def parse_load(source_item, classes):

    img_path, lable_info = source_item

    gt_bboxes = []
    gt_labels = []
    for obj in lable_info:
        bbox = obj['box']
        box = [
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[0] + bbox[2]),
            float(bbox[1] + bbox[3])
        ]
        gt_bboxes.append(box)
        if obj.get('tag') not in classes:
            continue

        gt_labels.append(int(classes.index(obj['tag'])))

    if len(gt_bboxes) == 0:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)

    img_info = {
        'gt_bboxes': np.array(gt_bboxes, dtype=np.float32),
        'gt_labels': np.array(gt_labels, dtype=np.int64),
        'filename': img_path,
    }

    return img_info


@DATASOURCES.register_module
class DetSourceCrowdHuman(DetSourceBase):
    CLASSES = ['mask', 'person']
    '''
    Citation:
        @article{shao2018crowdhuman,
        title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
        author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
        journal={arXiv preprint arXiv:1805.00123},
        year={2018}
    }

    '''
    """
    data dir is as follows:
    ```
    |- data
        |-annotation_train.odgt
        |-images
            |-273271,1a0d6000b9e1f5b7.jpg
            |-...

    ```
    Example1:
        data_source = DetSourceCrowdHuman(
            ann_file='/your/data/annotation_train.odgt',
            img_prefix='/your/data/images',
            classes=${CLASSES}
        )
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 gt_op='vbox',
                 classes=CLASSES,
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 parse_fn=parse_load,
                 num_processes=int(cpu_count() / 2),
                 **kwargs) -> None:
        """
        Args:
            ann_file (str): Path to the annotation file.
            img_prefix (str): Path to a directory where images are held.
            gt_op (str): vbox(visible box), fbox(full box), hbox(head box), defalut vbox
            classes(list): classes defalut=['mask', 'person']
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        self.gt_op = gt_op

        super(DetSourceCrowdHuman, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):

        assert os.path.exists(self.ann_file), f'{self.ann_file} is not exists'
        assert os.path.exists(
            self.img_prefix), f'{self.img_prefix} is not exists'

        imgs_path_list = []
        labels_list = []

        with io.open(self.ann_file, 'r') as t:
            lines = t.readlines()

            for img_info in lines:
                img_info = json.loads(img_info.strip('\n'))
                img_path = os.path.join(self.img_prefix,
                                        img_info['ID'] + '.jpg')
                if os.path.exists(img_path):
                    imgs_path_list.append(img_path)
                    labels_list.append([{
                        'box': label_info[self.gt_op],
                        'tag': label_info['tag']
                    } for label_info in img_info['gtboxes']])

        return list(zip(imgs_path_list, labels_list))
