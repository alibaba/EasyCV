#! -*- coding: utf8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import time
import unittest
import uuid

import numpy as np
import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist
from tests.ut_config import TMP_DIR_LOCAL
from torch import nn
from torch.utils.data import DataLoader

from easycv.file import io
from easycv.hooks.optimizer_hook import OptimizerHook
from easycv.hooks.sync_norm_hook import SyncNormHook
from easycv.runner import EVRunner
from easycv.utils import get_root_logger
from easycv.utils.test_util import dist_exec_wrapper


def _build_model():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
            self.bn = nn.BatchNorm2d(3)
            # DistributedDataParallel will sync to rank0 param
            nn.init.constant_(self.linear.weight, 1.0)
            nn.init.constant_(self.linear.bias, 1.0)
            nn.init.constant_(self.bn.weight, 1.0)
            nn.init.constant_(self.bn.bias, 1.0)

        def forward(self, x):
            x = self.linear(x)
            x = self.bn(x)
            return x

        def train_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze()[0][0][0])

        def val_step(self, x, optimizer, **kwargs):
            return dict(loss=self(x).squeeze()[0][0][0])

    return Model()


class SyncNormHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    @unittest.skipIf(torch.cuda.device_count() <= 1, 'distributed unittest')
    def test_sync_norm_hook(self):
        cur_file_name = os.path.splitext(os.path.basename(__file__))[0]
        python_path = os.path.dirname(os.path.abspath(__file__))
        cmd = 'python -c \"import %s; %s.SyncNormHookTest._run()\"' % (
            cur_file_name, cur_file_name)

        dist_exec_wrapper(cmd, nproc_per_node=2, python_path=python_path)

    @staticmethod
    def _run():
        init_dist(launcher='pytorch')
        rank, _ = get_dist_info()

        model = _build_model()
        model = MMDistributedDataParallel(
            model.cuda(),
            find_unused_parameters=True,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

        if rank == 0:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.2, momentum=0.9)

        work_dir = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        io.makedirs(work_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file)
        runner = EVRunner(
            model=model, work_dir=work_dir, optimizer=optimizer, logger=logger)

        optimizer_config = OptimizerHook()
        runner.register_optimizer_hook(optimizer_config)
        hook = SyncNormHook(no_aug_epochs=2, interval=1)
        runner.register_hook(hook)

        loader = DataLoader(torch.ones((2, 3, 4, 4)))
        runner.run([loader], [('train', 1)], 3)

        state_dict = runner.model.module.state_dict()
        assert state_dict['bn.weight'].detach().cpu().numpy().all(
        ) == np.asarray([4.4149, 1.0000, 1.0000]).all()
        assert state_dict['bn.bias'].detach().cpu().numpy().all(
        ) == np.asarray([-1.6745, 1.0000, 1.0000]).all()
        assert state_dict['bn.running_mean'].detach().cpu().numpy().all(
        ) == np.asarray([2.3428, 2.3428, 2.3428]).all()
        assert state_dict['bn.running_var'].detach().cpu().numpy().all(
        ) == np.asarray([45813.5469, 45813.5469, 45813.5469]).all()

        if rank == 0:
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
