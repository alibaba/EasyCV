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
from torch import distributed as dist

from easycv.datasets import build_dataloader
from easycv.file import io
from easycv.hooks.swav_hook import SWAVHook
from easycv.runner import EVRunner
from easycv.utils import get_root_logger
from easycv.utils.test_util import pseudo_dist_init


class DummyDataset(object):

    def __getitem__(self, idx):
        output = {'img': [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]}
        return output

    def __len__(self):
        return 4


def _build_model():
    from easycv.models import build_model
    num_crops = [1, 1]
    epoch_queue_starts = 1
    model = dict(
        type='SWAV',
        pretrained=None,
        train_preprocess=['randomGrayScale', 'gaussianBlur'],
        backbone=dict(
            type='ResNet',
            depth=18,
            in_channels=3,
            out_indices=[4],  # 0: conv-1, x: stage-x
            norm_cfg=dict(type='SyncBN')),
        # swav need  mulit crop ,doesn't support vit based model
        neck=dict(
            type='NonLinearNeckSwav',
            in_channels=512,
            hid_channels=512,
            out_channels=128,
            with_avg_pool=False),
        config=dict(
            # multi crop setting
            num_crops=num_crops,

            # swav setting
            crops_for_assign=[0, 1],
            epsilon=0.05,
            nmb_prototypes=3000,
            sinkhorn_iterations=3,
            temperature=0.1,

            # queue setting
            queue_length=3840,
            epoch_queue_starts=epoch_queue_starts))

    return build_model(model)


class SWAVHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_swav_hook(self):
        work_dir = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        io.makedirs(work_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file)

        model = _build_model()
        model = MMDataParallel(model, device_ids=[0]).cuda()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)
        runner = EVRunner(
            model=model, work_dir=work_dir, optimizer=optimizer, logger=logger)

        dump_path = os.path.join(work_dir, 'data')
        swav_hook = SWAVHook(gpu_batch_size=4, dump_path=dump_path)
        runner.register_hook(swav_hook)

        dataset = DummyDataset()
        dataloader = build_dataloader(
            dataset, imgs_per_gpu=4, workers_per_gpu=1)
        runner.data_loader = [dataloader]

        pseudo_dist_init()
        runner.run([dataloader], [('train', 1)], 2)
        self.assertIn('queue0.pth', io.listdir(dump_path))

        shutil.rmtree(work_dir, ignore_errors=True)
        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
