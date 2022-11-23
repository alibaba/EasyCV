# Copyright (c) OpenMMLab. All rights reserved.

import os
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .base import DetSourceBase


def parse_txt(source_item, classes):
    img_path, txt_path = source_item
    with io.open(txt_path, 'r') as t:
        label_lines = t.read().splitlines()
        num = int(label_lines[0])
        label_lines = label_lines[1:]
        assert len(label_lines) == num, ' number of boxes is not equal '
        gt_bboxes = []
        gt_labels = []
        for obj in label_lines:
            line = obj.split()
            cls_id = int(line[0]) - 1

            box = (float(line[1]), float(line[2]), float(line[3]),
                   float(line[4]))
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
class DetSourceWiderPerson(DetSourceBase):
    CLASSES = [
        'pedestrians', 'riders', 'partially-visible persons', 'ignore regions',
        'crowd'
    ]

    def __init__(self,
                 path,
                 classes=CLASSES,
                 img_root_path=None,
                 label_root_path=None,
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 img_suffix='.jpg',
                 label_suffix='.txt',
                 parse_fn=parse_txt,
                 num_processes=int(cpu_count() / 2),
                 **kwargs) -> None:

        self.path = path
        self.img_root_path = img_root_path
        self.label_root_path = label_root_path
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix
        super(DetSourceWiderPerson, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):
        assert os.path.exists(self.path), f'{self.path} is not exists'

        if not self.img_root_path:
            self.img_root_path = os.path.join(self.path, '..', 'Images')
        if not self.label_root_path:
            self.label_root_path = os.path.join(self.path, '..', 'Annotations')

        imgs_path_list = []
        labels_path_list = []

        with io.open(self.path, 'r') as t:
            id_lines = t.read().splitlines()
            for id_line in id_lines:
                img_id = id_line.strip()
                if img_id == '':
                    continue
                img_path = os.path.join(self.img_root_path,
                                        img_id + self.img_suffix)

                imgs_path_list.append(img_path)

                label_path = os.path.join(
                    self.label_root_path,
                    img_id + self.img_suffix + self.label_suffix)
                labels_path_list.append(label_path)

        return list(zip(imgs_path_list, labels_path_list))
