#! -*- coding: utf8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest

import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from torch import nn

from easycv.datasets import build_dataloader
from easycv.file import io
from easycv.hooks.ema_hook import EMAHook
from easycv.hooks.optimizer_hook import OptimizerHook
from easycv.runner import EVRunner
from easycv.utils import get_root_logger
from easycv.utils.test_util import get_tmp_dir


class DummyDataset(object):

    def __getitem__(self, idx):
        return torch.ones((5, 2))

    def __len__(self):
        return 3


def _build_model():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            nn.init.constant_(self.linear.bias, 1.0)
            nn.init.constant_(self.linear.weight, 2.0)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze()[0])

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze()[0])

    return Model()


class EMAHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_ema_hook(self):
        model = _build_model()
        model = MMDataParallel(model, device_ids=[0])

        optimizer = torch.optim.SGD(model.parameters(), lr=1, momentum=0.9)
        tmp_dir = get_tmp_dir()
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(tmp_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file)
        runner = EVRunner(
            model=model, work_dir=tmp_dir, optimizer=optimizer, logger=logger)

        optimizer_config = OptimizerHook()
        runner.register_optimizer_hook(optimizer_config)

        hook = EMAHook(decay=0.99)

        dataset = DummyDataset()
        dataloader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=1)

        runner.register_hook(hook)
        runner.run([dataloader], [('train', 1)], 2)

        state_dict = runner.model.state_dict()
        self.assertEqual(
            state_dict['module.linear.weight'].detach().cpu().numpy().all(),
            np.asarray([[-15.8297, -15.8297]]).all())
        self.assertEqual(
            state_dict['module.linear.bias'].detach().cpu().numpy().all(),
            np.asarray([-16.8297]).all())

        ema_state_dict = runner.ema.model.state_dict()
        self.assertEqual(
            ema_state_dict['linear.weight'].detach().cpu().numpy().all(),
            np.asarray([[-15.8158, -15.8158]]).all())
        self.assertEqual(
            ema_state_dict['linear.bias'].detach().cpu().numpy().all(),
            np.asarray([-16.8158]).all())

        io.rmtree(runner.work_dir)


if __name__ == '__main__':
    unittest.main()
