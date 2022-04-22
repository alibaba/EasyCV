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
from easycv.hooks.dino_hook import DINOHook
from easycv.runner import EVRunner
from easycv.utils.logger import get_root_logger


class DummyDataset(object):

    def __getitem__(self, idx):
        output = {'img': [torch.randn(3, 224, 224), torch.randn(3, 224, 224)]}
        return output

    def __len__(self):
        return 4


def _build_model():
    from easycv.models import build_model
    model = dict(
        type='DINO',
        pretrained=None,
        train_preprocess=[
            'randomGrayScale', 'gaussianBlur', 'solarize'
        ],  # 2+6 view, has different augment pipeline, dino is complex
        backbone=dict(
            type='PytorchImageModelWrapper',
            # deit(224)
            model_name='dynamic_deit_small_p16',
        ),
        # swav need  mulit crop ,doesn't support vit based model
        neck=dict(type='DINONeck', in_dim=384, out_dim=65536),
        config=dict(
            use_bn_in_head=False,
            norm_last_layer=True,
        ))

    return build_model(model)


class DINOHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_byol_hook(self):
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

        dino_hook = DINOHook()
        runner.register_hook(dino_hook)

        dataset = DummyDataset()
        dataloader = build_dataloader(
            dataset, imgs_per_gpu=2, workers_per_gpu=1)
        runner.data_loader = [dataloader]
        runner.run([dataloader], [('train', 1)], 1)
        self.assertEqual(runner.optimizer.param_groups[0]['weight_decay'],
                         0.22)

        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
