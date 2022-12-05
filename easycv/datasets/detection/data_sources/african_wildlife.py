# Copyright (c) OpenMMLab. All rights reserved.

import glob
import os
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .base import DetSourceBase, _load_image


def parse_txt(source_item, classes):
    img_path, txt_path = source_item
    with io.open(txt_path, 'r') as t:
        label_lines = t.read().splitlines()
        gt_bboxes = []
        gt_labels = []
        for obj in label_lines:
            line = obj.split()
            cls_id = classes.index(classes[int(line[0])])
            height, width, n = _load_image(img_path)['img_shape']
            line = [
                int(float(line[1]) * width),
                int(float(line[2]) * height),
                int(float(line[3]) * width / 2),
                int(float(line[4]) * height / 2)
            ]
            box = (float(line[0] - line[2]), float(line[1] - line[3]),
                   float(line[0] + line[2]), float(line[1] + line[3]))
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
class DetSourceAfricanWildlife(DetSourceBase):
    """
    data dir is as follows:
    ```
    |- data
        |-buffalo
            |-001.jpg
            |-001.txt
            |-...
        |-elephant
            |-001.jpg
            |-001.txt
            |-...
        |-rhino
            |-001.jpg
            |-001.txt
        |-...


    ```
    Example1:
        data_source = DetSourceAfricanWildlife(
            path='/your/data/',
            classes=${CLASSES},
        )

    """

    CLASSES = ['buffalo', 'elephant', 'rhino', 'zebra']

    def __init__(self,
                 path,
                 classes=CLASSES,
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 img_suffix='.jpg',
                 label_suffix='.txt',
                 parse_fn=parse_txt,
                 num_processes=int(cpu_count() / 2),
                 **kwargs) -> None:
        """
        Args:
            path: path of img id list file in root
            classes: classes list
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            img_suffix: suffix of image file
            label_suffix: suffix of label file
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.path = path
        self.img_suffix = img_suffix
        self.label_suffix = label_suffix

        super(DetSourceAfricanWildlife, self).__init__(
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
            img_list = glob.glob(os.path.join(img_path, '*' + self.img_suffix))
            for img in img_list:
                label_path = img.replace(self.img_suffix, self.label_suffix)
                assert os.path.exists(
                    label_path), f'{label_path} is not exists'
                imgs_path_list.append(img)
                labels_path_list.append(label_path)

        return list(zip(imgs_path_list, labels_path_list))
