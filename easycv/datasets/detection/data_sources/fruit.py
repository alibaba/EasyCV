# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
import xml.etree.ElementTree as ET
from multiprocessing import cpu_count

from easycv.datasets.registry import DATASOURCES
from .base import DetSourceBase
from .voc import parse_xml


@DATASOURCES.register_module
class DetSourceFruit(DetSourceBase):
    """
    data dir is as follows:
    ```
    |- data
        |-banana_2.jpg
        |-banana_2.xml
        |-...


    ```
    Example1:
        data_source = DetSourceFruit(
            path='/your/data/',
            classes=${CLASSES},

    """
    CLASSES = ['apple', 'banana', 'orange']

    def __init__(self,
                 path,
                 classes=CLASSES,
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
        super(DetSourceFruit, self).__init__(
            classes=classes,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly,
            parse_fn=parse_fn,
            num_processes=num_processes)

    def get_source_iterator(self):

        assert os.path.exists(self.path), f'{self.path} is not exists'
        imgs_path_list = []
        labels_path_list = []
        img_list = glob.glob(os.path.join(self.path, '*' + self.img_suffix))
        for img in img_list:
            label_path = img.replace(self.img_suffix, self.label_suffix)
            assert os.path.exists(label_path), f'{label_path} is not exists'
            imgs_path_list.append(img)
            labels_path_list.append(label_path)

        return list(zip(imgs_path_list, labels_path_list))
