# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import time

from PIL import Image, ImageFile

from easycv.datasets.registry import DATASOURCES
from easycv.file import io


@DATASOURCES.register_module
class SSLSourceImageList(object):
    """ datasource for classification

    Args:
        list_file : str / list(str), str means a input image list file path,
            this file contains records as  `image_path label` in list_file
            list(str) means multi image list, each one contains some records as `image_path label`
        root: str / list(str), root path for image_path, each list_file will need a root,
            if len(root) < len(list_file), we will use root[-1] to fill root list.
        max_try: int, max try numbers of reading image
    """

    def __init__(self, list_file, root='', max_try=20):

        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.max_try = max_try

        if isinstance(list_file, str):
            assert isinstance(root, str), 'list_file is str, root must be str'
            list_file = [list_file]
            root = [root]
        else:
            assert isinstance(list_file,
                              list), 'list_file should be str or list(str)'
            if isinstance(root, str):
                root = [root]
            if not isinstance(root, list):
                raise ValueError('root must be str or list(str), but get %s' %
                                 type(root))

            if len(root) < len(list_file):
                logging.warning(
                    'len(root) < len(list_file), fill root with root last!')
                root = root + [root[-1]] * (len(list_file) - len(root))

        self.fns = []
        for l, r in zip(list_file, root):
            fns = self.parse_list_file(l, r)
            self.fns += fns

    @staticmethod
    def parse_list_file(list_file, root):
        with io.open(list_file, 'r') as f:
            lines = f.readlines()

        fns = [l.strip() for l in lines]
        fns = [os.path.join(root, fn) for fn in fns]

        return fns

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

        return {'img': img}
