# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import random
import time

from PIL import Image, ImageFile

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from easycv.utils.dist_utils import dist_zero_exec
from .utils import split_listfile_byrank


@DATASOURCES.register_module
class ClsSourceImageListByClass(object):
    """
    Get the same `m_per_class` samples by the label idx.

    Args:
        list_file : str / list(str), str means a input image list file path,
            this file contains records as  `image_path label` in list_file
            list(str) means multi image list, each one contains some records as `image_path label`
        root: str / list(str), root path for image_path, each list_file will need a root.
        m_per_class: num of samples for each class.
        delimeter: str, delimeter of each line in the `list_file`
        split_huge_listfile_byrank: Adapt to the situation that the memory cannot fully load a huge amount of data list.
            If split, data list will be split to each rank.
        cache_path: if `split_huge_listfile_byrank` is true, cache list_file will be saved to cache_path.
        max_try: int, max try numbers of reading image
    """

    def __init__(self,
                 root,
                 list_file,
                 m_per_class=2,
                 delimeter=' ',
                 split_huge_listfile_byrank=False,
                 cache_path='data/',
                 max_try=20):

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        # TODO: support return list, donot save split file
        # TODO: support loading list_file that have already been split
        if split_huge_listfile_byrank:
            with dist_zero_exec():
                list_file = split_listfile_byrank(
                    list_file=list_file,
                    label_balance=True,
                    save_path=cache_path)

        with io.open(list_file, 'r') as f:
            lines = f.readlines()

        self.m_per_class = m_per_class
        self.has_labels = len(lines[0].split(delimeter)) >= 2
        assert self.has_labels is True

        label2files = {}

        for l in lines:
            label = int(l.strip().split(delimeter)[1])
            path = l.strip().split(delimeter)[0]

            if label in label2files.keys():
                label2files[label].append(path)
            else:
                label2files[label] = [path]

        self.labels = list(label2files.keys())
        self.fns_by_labels = [label2files[i] for i in self.labels]
        self.root = root

        self.initialized = False
        self.max_try = max_try

    def get_length(self):
        return len(self.fns_by_labels)

    def get_sample(self, idx):
        label = self.labels[idx]
        image_list = self.fns_by_labels[idx]
        if len(image_list) < 1:
            logging.info('%s :image list contain < 1 image' % idx)
            return self.get_sample(self, idx + 1)

        if self.m_per_class > len(image_list):
            image_list = int(self.m_per_class / len(image_list) +
                             1) * image_list

        sample_list = random.sample(image_list, self.m_per_class)
        return_img = []
        return_label = []

        for path in sample_list:
            img = None
            try_idx = 0
            while not img and try_idx < self.max_try:
                try:
                    img = Image.open(
                        io.open(os.path.join(self.root, path), 'rb'))
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                except:
                    logging.warning('Try read file fault, %s' %
                                    os.path.join(self.root, path))
                    time.sleep(1)
                    img = None

                try_idx += 1

            if img is None:
                return self.get_sample(idx + 1)

            return_img.append(img)
            return_label.append(label)

        result_dict = {'img': return_img, 'gt_labels': return_label}
        return result_dict
