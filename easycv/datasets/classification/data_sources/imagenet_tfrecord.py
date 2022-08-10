# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

from PIL import ImageFile

from easycv.datasets.registry import DATASOURCES
from easycv.datasets.utils.tfrecord_util import (download_tfrecord,
                                                 get_path_and_index)
from easycv.file import io
from easycv.file.utils import is_oss_path
from easycv.utils import dist_utils


@DATASOURCES.register_module
class ClsSourceImageNetTFRecord(object):
    """ data source for imagenet tfrecord.
    """

    def __init__(self,
                 list_file='',
                 root='',
                 file_pattern=None,
                 cache_path='data/cache/',
                 max_try=10):
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        self.max_try = max_try

        if file_pattern:
            assert (not list_file) and (
                not root), 'only support one of list_file and file_pattern'
            file_list = io.glob(file_pattern)
            is_oss = True if is_oss_path(file_pattern) else False
        else:
            with io.open(list_file, 'r') as f:
                lines = f.readlines()
                file_list = [os.path.join(root, i.strip()) for i in lines]
            is_oss = True if is_oss_path(list_file) else False

        if is_oss:
            local_size = dist_utils.get_num_gpu_per_node()
            local_rank = dist_utils.local_rank()
            logging.info('Strat download oss data to target_path!')
            self.data_list, self.index_list = download_tfrecord(
                file_list,
                cache_path,
                slice_count=local_size,
                slice_id=local_rank,
                force=False)
            logging.info('Finished download oss data!')
        else:
            self.data_list, self.index_list = get_path_and_index(file_list)

    def __len__(self):
        return len(self.path_list)
