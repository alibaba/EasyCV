#! -*- coding: utf8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import time
import unittest
import uuid

import torch
from mmcv.parallel import MMDataParallel
from tests.ut_config import TMP_DIR_LOCAL

from easycv.datasets import build_dataloader
from easycv.file import io
from easycv.hooks.byol_hook import BYOLHook
from easycv.runner import EVRunner
from easycv.utils import get_root_logger


class DummyDataset(object):

    def __getitem__(self, idx):
        output = {'img': [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]}
        return output

    def __len__(self):
        return 4


def _build_model():
    from easycv.models import build_model
    model = dict(
        type='BYOL',
        pretrained=None,
        base_momentum=0.996,
        backbone=dict(
            type='ResNet',
            depth=18,
            in_channels=3,
            out_indices=[4],  # 0: conv-1, x: stage-x
            norm_cfg=dict(type='BN')),
        neck=dict(
            type='NonLinearNeckV2',
            in_channels=512,
            hid_channels=1024,
            out_channels=256,
            with_avg_pool=True),
        head=dict(
            type='LatentPredictHead',
            size_average=True,
            predictor=dict(
                type='NonLinearNeckV2',
                in_channels=256,
                hid_channels=1024,
                out_channels=256,
                with_avg_pool=False)))

    return build_model(model)


class BYOLHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_byol_hook(self):
        work_dir = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        io.makedirs(work_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file)

        model = _build_model()
        model = MMDataParallel(model, device_ids=[0])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)
        runner = EVRunner(
            model=model, work_dir=work_dir, optimizer=optimizer, logger=logger)

        byol_hook = BYOLHook()
        runner.register_hook(byol_hook)

        dataset = DummyDataset()
        dataloader = build_dataloader(
            dataset, imgs_per_gpu=2, workers_per_gpu=1)
        runner.run([dataloader], [('train', 1)], 1)
        self.assertEqual(runner.model.module.momentum, 0.998)

        runner.run([dataloader], [('train', 1)], 2)
        self.assertEqual(format(runner.model.module.momentum, '.4f'), '0.9994')

        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
