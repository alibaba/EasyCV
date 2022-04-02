#! -*- coding: utf8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest
import uuid

import torch
from mmcv import Config
from torch import nn
from torch.utils.data import DataLoader

from easycv.file import io
from easycv.hooks.export_hook import ExportHook
from easycv.models.registry import MODELS
from easycv.runner import EVRunner
from easycv.utils import get_root_logger
from easycv.utils.test_util import get_tmp_dir


class ExportHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_export_hook(self):
        model = _build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)
        log_config = dict(
            interval=1,
            hooks=[
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook'),
            ])
        checkpoint_config = dict(interval=1)

        work_dir = get_tmp_dir()
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file)
        runner = EVRunner(
            model=model, work_dir=work_dir, optimizer=optimizer, logger=logger)

        runner.register_logger_hooks(log_config)
        runner.register_checkpoint_hook(checkpoint_config)

        cfg = Config(cfg_dict=dict(model=dict(type='TestExportModel')))
        cfg.work_dir = work_dir
        hook = ExportHook(cfg)
        loader = DataLoader(torch.ones((3, 2)))

        runner.register_hook(hook)
        runner.run([loader], [('train', 1)], 2)
        files_list = io.listdir(work_dir)
        self.assertIn('epoch_2_export.pt', files_list)
        self.assertIn('epoch_1.pth', files_list)
        self.assertIn('epoch_2.pth', files_list)

        io.rmtree(work_dir)


def _build_model():

    @MODELS.register_module()
    class TestExportModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze())

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze())

    return TestExportModel()


if __name__ == '__main__':
    unittest.main()
