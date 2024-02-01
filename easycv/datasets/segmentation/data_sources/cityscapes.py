# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os
import subprocess

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from easycv.file.image import load_image as _load_img
from .raw import SegSourceRaw

try:
    import cityscapesscripts.helpers.labels as CSLabels
except ModuleNotFoundError as e:
    res = subprocess.call('pip install cityscapesscripts', shell=True)
    if res != 0:
        info_string = (
            '\n\nAuto install failed! Please install cityscapesscripts with the following commands :\n'
            '\t`pip install cityscapesscripts`\n')
        raise ModuleNotFoundError(info_string)


def load_seg_map_cityscape(seg_path, reduce_zero_label):
    gt_semantic_seg = _load_img(seg_path, mode='P')
    gt_semantic_seg_copy = gt_semantic_seg.copy()
    for labels in CSLabels.labels:
        gt_semantic_seg_copy[gt_semantic_seg == labels.id] = labels.trainId

    return {'gt_semantic_seg': gt_semantic_seg_copy}


@DATASOURCES.register_module
class SegSourceCityscapes(SegSourceRaw):
    """Cityscapes datasource
    """
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self,
                 img_suffix='_leftImg8bit.png',
                 label_suffix='_gtFine_labelIds.png',
                 **kwargs):
        super(SegSourceCityscapes, self).__init__(
            img_suffix=img_suffix, label_suffix=label_suffix, **kwargs)

    def __getitem__(self, idx):
        result_dict = self.samples_list[idx]
        load_success = True
        try:
            # avoid data cache from taking up too much memory
            if not self.cache_at_init and not self.cache_on_the_fly:
                result_dict = copy.deepcopy(result_dict)

            if not self.cache_at_init:
                if result_dict.get('img', None) is None:
                    img = _load_img(result_dict['filename'], mode='BGR')
                    result = {
                        'img': img.astype(np.float32),
                        'img_shape': img.shape,  # h, w, c
                        'ori_shape': img.shape,
                    }
                    result_dict.update(result)
                if result_dict.get('gt_semantic_seg', None) is None:
                    result_dict.update(
                        load_seg_map_cityscape(
                            result_dict['seg_filename'],
                            reduce_zero_label=self.reduce_zero_label))
                if self.cache_on_the_fly:
                    self.samples_list[idx] = result_dict
            result_dict = self.post_process_fn(copy.deepcopy(result_dict))
            self._retry_count = 0
        except Exception as e:
            logging.warning(e)
            load_success = False

        if not load_success:
            logging.warning(
                'Something wrong with current sample %s,Try load next sample...'
                % result_dict.get('filename', ''))
            self._retry_count += 1
            if self._retry_count >= self._max_retry_num:
                raise ValueError('All samples failed to load!')

            result_dict = self[(idx + 1) % self.num_samples]

        return result_dict

    def get_source_iterator(self):

        self.img_files = [
            os.path.join(self.img_root, i)
            for i in io.listdir(self.img_root, recursive=True)
            if i.endswith(self.img_suffix[0])
        ]

        self.label_files = []
        for img_path in self.img_files:
            self.img_root = os.path.join(self.img_root, '')
            img_name = img_path.replace(self.img_root,
                                        '')[:-len(self.img_suffix[0])]
            find_label_path = False
            for label_format in self.label_suffix:
                label_path = os.path.join(self.label_root,
                                          img_name + label_format)
                if io.exists(label_path):
                    find_label_path = True
                    self.label_files.append(label_path)
                    break
            if not find_label_path:
                logging.warning(
                    'Not find label file %s for img: %s, skip the sample!' %
                    (label_path, img_path))
                self.img_files.remove(img_path)

        assert len(self.img_files) == len(self.label_files)
        assert len(
            self.img_files) > 0, 'No samples found in %s' % self.img_root

        return list(zip(self.img_files, self.label_files))
