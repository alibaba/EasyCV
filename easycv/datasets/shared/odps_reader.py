# Copyright (c) Alibaba, Inc. and its affiliates.
import base64
import os
import time
from random import randint

import numpy as np
import requests
from mmcv.runner import get_dist_info
from PIL import Image, ImageFile

from easycv.datasets.registry import DATASOURCES
from easycv.file import io

ImageFile.LOAD_TRUNCATED_IMAGES = True

data_cache = {}

SUPPORT_IMAGE_TYPE = ['url', 'base64']
DATALOADER_WORKID = -1
DATALOADER_WORKNUM = 1


def set_dataloader_workid(value):
    global DATALOADER_WORKID
    DATALOADER_WORKID = value


def set_dataloader_worknum(value):
    global DATALOADER_WORKNUM
    DATALOADER_WORKNUM = value


def get_dist_image(img_url, max_try=10):
    img = None
    try_idx = 0
    while try_idx < max_try:
        try:
            # http url
            if img_url.startswith('http'):
                img = Image.open(requests.get(img_url,
                                              stream=True).raw).convert('RGB')
            # oss url
            else:
                img = Image.open(io.open(img_url, 'rb')).convert('RGB')
        except:
            print('Try read file fault, %s' % img_url)
            time.sleep(1)
            img = None
        try_idx += 1
        if img is not None:
            break
    return img


@DATASOURCES.register_module
class OdpsReader(object):

    def __init__(self,
                 table_name,
                 selected_cols=[],
                 excluded_cols=[],
                 random_start=False,
                 odps_io_config=None,
                 image_col=['url_image'],
                 image_type=['url']):
        """Init odps reader and datasource set to load data from odps table

        Args:
            table_name (str): odps table to load
            selected_cols (list(str)):  select column
            excluded_cols (list(str)):  exclude column
            random_start (bool):  random start for odps table
            odps_io_config (dict):  odps config contains access_id, access_key, endpoint
            image_col (list(str)):  image column names
            image_type (list(str)):  image column types support url/base64, must be same length with image type or 0
        Returns :
            None
        """

        assert (odps_io_config
                is not None), 'odps_io_config should be set for OdpsReader !'
        # set odps config
        if odps_io_config is not None:
            assert 'access_id' in odps_io_config.keys(
            ), 'odps_io_config should contains access_id'
            assert 'access_key' in odps_io_config.keys(
            ), 'odps_io_config should contains access_key'
            assert 'end_point' in odps_io_config.keys(
            ), 'odps_io_config should contains end_point'

            # distributed env, especially on PAI-Studio.
            if not os.path.exists('.odps_io_config'):
                write_idx = 0
                while not os.path.exists('.odps_io_config') and write_idx < 10:
                    write_idx += 1
                    try:
                        with open('.odps_io_config', 'w') as f:
                            f.write('access_id=%s\n' %
                                    (odps_io_config['access_id']))
                            f.write('access_key=%s\n' %
                                    (odps_io_config['access_key']))
                            f.write('end_point=%s\n' %
                                    (odps_io_config['end_point']))
                    except:
                        pass

            os.environ['ODPS_CONFIG_FILE_PATH'] = '.odps_config'

        # set distribute read
        rank, world_size = get_dist_info()

        # there are two multi process world for dataset, first multi-gpu worker, secord multi process for per GPU
        self.dataloader_init = False

        # keep input args
        assert (
            type(image_type) == list and type(image_col) == list
        ), 'image_col, image_type for OdpsReader must be set as list of (column name), list of (image type)'
        assert (len(image_type) == len(image_col))

        self.selected_cols = selected_cols
        self.excluded_cols = excluded_cols
        self.rank = rank
        self.ddp_world_size = world_size
        self.table_name = table_name
        self.random_start = random_start

        # init for reader
        import common_io

        self.reader = common_io.table.TableReader(
            self.table_name,
            slice_id=self.rank,
            slice_count=self.ddp_world_size,
            selected_cols=','.join(self.selected_cols),
            excluded_cols=','.join(self.excluded_cols),
        )

        self.length = self.reader.get_row_count()
        self.world_size = self.ddp_world_size
        if self.random_start:
            self.idx = randint(0, self.length)
            self.reader.seek(self.idx)
        else:
            self.idx = 0

        # init for find image
        self.schema = self.reader.get_schema()
        self.schema_name = [i[0] for i in self.schema]
        # find base64 image in odps schema
        self.base64_image_idx = []
        self.url_image_idx = []

        for idx, s in enumerate(self.schema):
            if s[0] in image_col:
                assert (
                    s[1] == 'string'
                ), 'ODPS image column must be string type, %s is %s !' % (s[0],
                                                                          s[1])
                idx_type = image_type[image_col.index(s[0])]
                assert (
                    idx_type in SUPPORT_IMAGE_TYPE
                ), 'image_type must set in support image type : url / base64'
                if idx_type == 'url':
                    self.url_image_idx.append(idx)
                if idx_type == 'base64':
                    self.base64_image_idx.append(idx)

        delattr(self, 'reader')
        return

    def get_length(self):
        return self.length * self.world_size

    def reset_reader(self, dataloader_workid, dataloader_worknum):
        import common_io

        self.reader = common_io.table.TableReader(
            self.table_name,
            slice_id=self.rank * dataloader_worknum + dataloader_workid,
            slice_count=self.ddp_world_size * dataloader_worknum,
            selected_cols=','.join(self.selected_cols),
            excluded_cols=','.join(self.excluded_cols),
        )

        self.length = self.reader.get_row_count()
        self.world_size = self.ddp_world_size * dataloader_worknum
        if self.random_start:
            self.idx = randint(0, self.length)
            self.reader.seek(self.idx)
        else:
            self.idx = 0

    def get_sample(self, idx):
        global DATALOADER_WORKID
        global DATALOADER_WORKNUM

        # we must del reader before init to support pytorch dataloader multi-process
        if not hasattr(self, 'reader'):
            import common_io

            self.reader = common_io.table.TableReader(
                self.table_name,
                slice_id=self.rank,
                slice_count=self.ddp_world_size,
                selected_cols=','.join(self.selected_cols),
                excluded_cols=','.join(self.excluded_cols),
            )

        if not self.dataloader_init:
            # num_per_gpu = 1 means we should not split reader
            if DATALOADER_WORKNUM == 1:
                self.dataloader_init = True
            elif DATALOADER_WORKNUM < 1:
                print('num_per_gpu for OdpsReader should >= 1')
            else:
                # if DATALOADER_WORKID == -1:
                assert (
                    DATALOADER_WORKID > -1
                ), "num_per_gpu for OdpsReader > 1, but DATALOADER_WORKNUM didn't be set by work_fn, False"
                self.reset_reader(DATALOADER_WORKID, DATALOADER_WORKNUM)
                self.dataloader_init = True

        self.idx += 1
        t = self.reader.read()[0]

        if self.idx == self.length:
            self.reader.seek(-self.length)
            self.idx = 0

        return_dict = {}
        # need set oss_io before
        for idx, m in enumerate(t):
            if idx in self.base64_image_idx:
                return_dict[self.schema_name[idx]] = Image.fromarray(
                    np.frombuffer(self.b64_decode(m)))
            elif idx in self.url_image_idx:
                return_dict[self.schema_name[idx]] = get_dist_image(m, 5)
            else:
                return_dict[self.schema_name[idx]] = m

        return return_dict

    def b64_decode(string):
        return base64.decodebytes(string.encode())
