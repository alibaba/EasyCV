# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import time

from PIL import Image, ImageFile

from easycv.datasets.registry import DATASOURCES
from easycv.file import io
from easycv.utils.dist_utils import dist_zero_exec
from .utils import split_listfile_byrank


@DATASOURCES.register_module
class ClsSourceImageList(object):
    """ data source for classification

        Args:
            list_file : str / list(str), str means a input image list file path,
                this file contains records as  `image_path label` in list_file
                list(str) means multi image list, each one contains some records as `image_path label`
            root: str / list(str), root path for image_path, each list_file will need a root,
                if len(root) < len(list_file), we will use root[-1] to fill root list.
            delimeter: str, delimeter of each line in the `list_file`
            split_huge_listfile_byrank: Adapt to the situation that the memory cannot fully load a huge amount of data list.
                If split, data list will be split to each rank.
            split_label_balance: if `split_huge_listfile_byrank` is true, whether split with label balance
            cache_path: if `split_huge_listfile_byrank` is true, cache list_file will be saved to cache_path.
            max_try: int, max try numbers of reading image
    """

    def __init__(self,
                 list_file,
                 root='',
                 delimeter=' ',
                 split_huge_listfile_byrank=False,
                 split_label_balance=False,
                 cache_path='data/',
                 max_try=20):

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.max_try = max_try

        # DistributedMPSampler need this attr
        self.has_labels = True

        if isinstance(list_file, str):
            assert isinstance(root, str), 'list_file is str, root must be str'
            list_file = [list_file]
            root = [root]
        else:
            assert isinstance(list_file, list), \
                'list_file should be str or list(str)'
            root = [root] if isinstance(root, str) else root
            if not isinstance(root, list):
                raise ValueError('root must be str or list(str), but get %s' %
                                 type(root))

            if len(root) < len(list_file):
                logging.warning(
                    'len(root) < len(list_file), fill root with root last!')
                root = root + [root[-1]] * (len(list_file) - len(root))

        # TODO: support return list, donot save split file
        # TODO: support loading list_file that have already been split
        if split_huge_listfile_byrank:
            with dist_zero_exec():
                list_file = split_listfile_byrank(
                    list_file=list_file,
                    label_balance=split_label_balance,
                    save_path=cache_path)

        self.fns = []
        self.labels = []
        for l, r in zip(list_file, root):
            fns, labels = self.parse_list_file(l, r, delimeter)
            self.fns += fns
            self.labels += labels

    @staticmethod
    def parse_list_file(list_file, root, delimeter):
        with io.open(list_file, 'r') as f:
            lines = f.readlines()

        fns = []
        labels = []

        for l in lines:
            splits = l.strip().split(delimeter)
            fns.append(os.path.join(root, splits[0]))
            # must be int,other with mmcv collect will crash
            label = [int(i) for i in splits[1:]]
            labels.append(
                label[0]) if len(label) == 1 else labels.append(label)

        return fns, labels

    def get_length(self):
        return len(self.fns)

    def get_sample(self, idx):
        img = None
        try_idx = 0

        while img is None and try_idx < self.max_try:
            try:
                img = Image.open(io.open(self.fns[idx], 'rb'))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
            except:
                # frequent access to oss will cause error, sleep can aviod it
                time.sleep(1)
                logging.warning('Try read file fault, %s' % self.fns[idx])
                img = None

            try_idx += 1

        if img is None:
            return self.get_sample(idx + 1)

        label = self.labels[idx]

        result_dict = {'img': img, 'gt_labels': label}
        return result_dict
