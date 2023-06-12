#! -*- coding: utf8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest
import uuid

import torch
from mmcv.utils import get_logger
from tests.ut_config import TMP_DIR_OSS
from torch import nn
from torch.utils.data import DataLoader

from easycv.file import io
from easycv.hooks.oss_sync_hook import OSSSyncHook
from easycv.runner import EVRunner
from easycv.utils.test_util import get_tmp_dir


class OSSSyncHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        io.access_oss()

    def test_oss_sync_hook(self):
        model = _build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)
        log_config = dict(
            interval=1,
            hooks=[
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook'),
            ])
        checkpoint_config = dict(interval=1)

        tmp_dir = get_tmp_dir()
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(tmp_dir, '{}.log'.format(timestamp))
        # use random name, fix the logger settings of other unittest cause the current logger settings invalid
        logger = get_logger(name=uuid.uuid4().hex, log_file=log_file)
        runner = EVRunner(
            model=model, work_dir=tmp_dir, optimizer=optimizer, logger=logger)

        runner.register_logger_hooks(log_config)
        runner.register_checkpoint_hook(checkpoint_config)

        oss_work_dir = os.path.join(TMP_DIR_OSS, uuid.uuid4().hex)
        hook = OSSSyncHook(
            runner.work_dir,
            oss_work_dir,
            other_file_list=['**/events.out.tfevents*', '**/*log*'])
        loader = DataLoader(torch.ones((5, 2)))

        runner.register_hook(hook)
        runner.run([loader], [('train', 1)], 1)

        # sleep to wait for oss
        time.sleep(1)

        self.assertTrue(io.exists(os.path.join(oss_work_dir, 'epoch_1.pth')))
        self.assertTrue(
            io.exists(os.path.join(oss_work_dir, '%s.log' % timestamp)))
        self.assertTrue(
            io.exists(os.path.join(oss_work_dir, '%s.log.json' % timestamp)))
        self.assertTrue(io.exists(os.path.join(oss_work_dir, 'tf_logs/')))
        events_file = io.glob(
            os.path.join(oss_work_dir, 'tf_logs/events.out.tfevents.*'))
        self.assertTrue(len(events_file) >= 1)
        io.rmtree(oss_work_dir)
        io.rmtree(runner.work_dir)


def _build_model():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze())

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze())

    return Model()


if __name__ == '__main__':
    unittest.main()
