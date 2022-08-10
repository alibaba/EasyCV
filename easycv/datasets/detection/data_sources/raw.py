# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import logging
import os
from multiprocessing import cpu_count

import numpy as np

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from easycv.utils.bbox_util import batched_cxcywh2xyxy_with_shape
from .base import DetSourceBase

img_formats = ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.dng']
label_formats = ['.txt']


def parse_raw(source_iter, classes=None, delimeter=' '):
    img_path, label_path = source_iter

    source_info = {'filename': img_path}

    with io.open(label_path, 'r') as f:
        labels_and_boxes = np.array(
            [line.split(delimeter) for line in f.read().splitlines()])

    if not len(labels_and_boxes):
        return source_info

    labels = labels_and_boxes[:, 0]
    bboxes = labels_and_boxes[:, 1:]

    source_info.update({
        'gt_bboxes': np.array(bboxes, dtype=np.float32),
        'gt_labels': labels.astype(np.int64)
    })

    return source_info


@DATASOURCES.register_module
class DetSourceRaw(DetSourceBase):
    """
    data dir is as follows:
    ```
    |- data_dir
        |-images
            |-1.jpg
            |-...
        |-labels
            |-1.txt
            |-...

    ```
    Label txt file is as follows:
    The first column is the label id, and columns 2 to 5 are
    coordinates relative to the image width and height [x_center, y_center, bbox_w, bbox_h].
    ```
    15 0.519398 0.544087 0.476359 0.572061
    2 0.501859 0.820726 0.996281 0.332178
    ...
    ```
    Example:
        data_source = DetSourceRaw(
            img_root_path='/your/data_dir/images',
            label_root_path='/your/data_dir/labels',
        )
    """

    def __init__(self,
                 img_root_path,
                 label_root_path,
                 classes=[],
                 cache_at_init=False,
                 cache_on_the_fly=False,
                 delimeter=' ',
                 parse_fn=parse_raw,
                 num_processes=int(cpu_count() / 2),
                 **kwargs):
        """
        Args:
            img_root_path: images dir path
            label_root_path: labels dir path
            classes(list, optional): classes list
            cache_at_init: if set True, will cache in memory in __init__ for faster training
            cache_on_the_fly: if set True, will cache in memroy during training
            delimeter: delimeter of txt file
            parse_fn: parse function to parse item of source iterator
            num_processes: number of processes to parse samples
        """

        self.delimeter = delimeter
        self.img_root_path = img_root_path
        self.label_root_path = label_root_path

        parse_fn = functools.partial(parse_fn, delimeter=delimeter)

        super(DetSourceRaw, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):
        self.img_files = [
            os.path.join(self.img_root_path, i)
            for i in io.listdir(self.img_root_path, recursive=True)
            if os.path.splitext(i)[-1].lower() in img_formats
        ]

        self.label_files = []
        for img_path in self.img_files:
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            find_label_path = False
            for label_format in label_formats:
                lable_path = os.path.join(self.label_root_path,
                                          img_name + label_format)
                if io.exists(lable_path):
                    find_label_path = True
                    self.label_files.append(lable_path)
                    break
            if not find_label_path:
                logging.warning(
                    'Not find label file %s for img: %s, skip the sample!' %
                    (lable_path, img_path))
                self.img_files.remove(img_path)

        assert len(self.img_files) == len(self.label_files)
        assert len(
            self.img_files) > 0, 'No samples found in %s' % self.img_root_path

        return list(zip(self.img_files, self.label_files))

    def post_process_fn(self, result_dict):
        result_dict = super(DetSourceRaw, self).post_process_fn(result_dict)

        result_dict['gt_bboxes'] = batched_cxcywh2xyxy_with_shape(
            result_dict['gt_bboxes'], shape=result_dict['img_shape'][:2])
        return result_dict
