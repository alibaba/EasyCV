#! -*- coding: utf8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import shutil
import time
import unittest
import uuid

import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from tests.ut_config import (COCO_CLASSES, DET_DATA_SMALL_COCO_LOCAL,
                             IMG_NORM_CFG_255, TMP_DIR_LOCAL)
from torch import nn

from easycv.datasets import build_dataloader
from easycv.datasets.builder import build_dataset
from easycv.file import io
from easycv.hooks.sync_random_size_hook import SyncRandomSizeHook
from easycv.runner import EVRunner
from easycv.utils.logger import get_root_logger
from easycv.utils.test_util import dist_exec_wrapper


def _build_model():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(160, 4)
            self.bn = nn.BatchNorm2d(3)

        def forward(self, x):
            x = self.linear(x)
            x = self.bn(x)
            return x

        def train_step(self, x, optimizer, **kwargs):
            img = x['img']
            output = self(img)
            return dict(loss=torch.sum(output))

        def val_step(self, x, optimizer, **kwargs):
            img = x['img']
            output = self(img)
            return dict(loss=torch.sum(output))

    return Model()


def _build_dataset():
    img_scale = (160, 160)

    pipeline = [
        dict(type='MMResize', img_scale=img_scale, keep_ratio=True),
        dict(type='MMPad', pad_to_square=True, pad_val=(114.0, 114.0, 114.0)),
        dict(type='MMNormalize', **IMG_NORM_CFG_255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
    ]

    dataset_cfg = dict(
        type='DetImagesMixDataset',
        data_source=dict(
            type='DetSourceCoco',
            ann_file=os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                                  'instances_train2017_20.json'),
            img_prefix=os.path.join(DET_DATA_SMALL_COCO_LOCAL, 'train2017'),
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=COCO_CLASSES,
            filter_empty_gt=False,
            iscrowd=False),
        pipeline=pipeline,
        dynamic_scale=img_scale)

    return build_dataset(dataset_cfg)


class SyncRandomSizeHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    @unittest.skipIf(torch.cuda.device_count() <= 1, 'distributed unittest')
    def test_sync_random_size_hook(self):
        cur_file_name = os.path.splitext(os.path.basename(__file__))[0]
        python_path = os.path.dirname(os.path.abspath(__file__))
        cmd = 'python -c \"import %s; %s.SyncRandomSizeHookTest._run()\"' % (
            cur_file_name, cur_file_name)

        dist_exec_wrapper(cmd, nproc_per_node=2, python_path=python_path)

    @staticmethod
    def _run():
        init_dist(launcher='pytorch')
        rank, _ = get_dist_info()
        random.seed(2022)

        work_dir = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        io.makedirs(work_dir)

        model = _build_model()
        model = MMDistributedDataParallel(
            model.cuda(),
            find_unused_parameters=True,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file)
        runner = EVRunner(
            model=model, work_dir=work_dir, optimizer=optimizer, logger=logger)

        hook = SyncRandomSizeHook(
            ratio_range=(14, 26), img_scale=(160, 160), interval=1)
        runner.register_hook(hook)

        data_loader = build_dataloader(
            _build_dataset(), imgs_per_gpu=2, workers_per_gpu=1)
        runner.run([data_loader], [('train', 1)], 1)

        assert runner.data_loader.dataset._dynamic_scale == (576, 576)

        if rank == 0:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
