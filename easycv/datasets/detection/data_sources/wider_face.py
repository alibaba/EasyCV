# Copyright (c) OpenMMLab. All rights reserved.

import os
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from .base import DetSourceBase


def parse_load(source_item, classes):

    img_path, lable_info = source_item
    class_index, lable_bbox_info = lable_info

    gt_bboxes = []
    gt_labels = []
    for obj in lable_bbox_info:
        obj = obj.strip().split()
        box = [
            float(obj[0]),
            float(obj[1]),
            float(obj[0] + obj[2]),
            float(obj[1] + obj[3])
        ]
        gt_bboxes.append(box)
        gt_labels.append(int(obj[class_index]))

    if len(gt_bboxes) == 0:
        gt_bboxes = np.zeros((0, 4), dtype=np.float32)

    img_info = {
        'gt_bboxes': np.array(gt_bboxes, dtype=np.float32),
        'gt_labels': np.array(gt_labels, dtype=np.int64),
        'filename': img_path,
    }

    return img_info


@DATASOURCES.register_module
class DetSourceWiderFace(DetSourceBase):
    CLASSES = dict(
        blur=['clear', 'normal blur', 'heavy blur'],
        expression=['typical expression', 'exaggerate expression'],
        illumination=['normal illumination', 'extreme illumination'],
        occlusion=['no occlusion', 'partial occlusion', 'heavy occlusion'],
        pose=['typical pose', 'atypical pose'],
        invalid=['false valid image)', 'true (invalid image)'])
    '''
    Citation:
        @inproceedings{yang2016wider,
        Author = {Yang, Shuo and Luo, Ping and Loy, Chen Change and Tang, Xiaoou},
        Booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        Title = {WIDER FACE: A Face Detection Benchmark},
        Year = {2016}}

    '''
    """
    data dir is as follows:
    ```
    |- data
        |-wider_face_split
            |- wider_face_train_bbx_gt.txt
            |-...
        |-WIDER_train
            |-images
                |-0--Parade
                    |-0_Parade_marchingband_1_656.jpg
                    |...
                |- 24--Soldier_Firing
                |-...
        |-WIDER_test
            |-images
                |-0--Parade
                    |-0_Parade_marchingband_1_656.jpg
                    |...
                |- 24--Soldier_Firing
                |-...
        |-WIDER_val
            |-images
                |-0--Parade
                    |-0_Parade_marchingband_1_656.jpg
                    |...
                |- 24--Soldier_Firing
                |-...

    ```
    Example1:
        data_source = DetSourceWiderFace(
            ann_file='/your/data/wider_face_split/wider_face_train_bbx_gt.txt',
            img_prefix='/your/data/WIDER_train/images',
            classes=${class_option}
        )
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 classes='blur',
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 parse_fn=parse_load,
                 num_processes=int(cpu_count() / 2),
                 **kwargs) -> None:
        """
        Args:
            ann_file (str): Path to the annotation file.
            img_prefix (str): Path to a directory where images are held.
            classes(str): classes defalut='blur'
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.ann_file = ann_file
        self.img_prefix = img_prefix
        assert self.ann_file.endswith('.txt'), 'Only support `.txt` now!'
        assert isinstance(
            classes, str) and classes in self.CLASSES, 'class values is error'
        self.class_option = classes
        classes = self.CLASSES.get(classes)
        super(DetSourceWiderFace, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):
        class_index = dict(
            blur=4,
            expression=5,
            illumination=6,
            invalid=7,
            occlusion=8,
            pose=9)
        assert os.path.exists(self.ann_file), f'{self.ann_file} is not exists'
        assert os.path.exists(
            self.img_prefix), f'{self.img_prefix} is not exists'

        imgs_path_list = []
        labels_list = []

        last_index = 0

        def load_lable_info(img_info):
            imgs_path_list.append(
                os.path.join(self.img_prefix, img_info[0].strip()))
            lable_info = img_info[2:]
            if int(img_info[1]) != len(img_info[2:]):
                return
            labels_list.append((class_index[self.class_option], lable_info))

        with io.open(self.ann_file, 'r') as t:
            txt_label = t.read().splitlines()

            for i, _ in enumerate(txt_label[1:]):
                if '/' in _:
                    load_lable_info(txt_label[last_index:i + 1])
                    last_index = i + 1
            load_lable_info(txt_label[last_index:])

        return list(zip(imgs_path_list, labels_list))
