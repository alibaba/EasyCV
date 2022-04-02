#! -*- coding: utf8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest
import uuid

import torch
from tests.ut_config import TMP_DIR_LOCAL
from torch import nn
from torch.utils.data import DataLoader

from easycv.core.evaluation.builder import build_evaluator
from easycv.file import io
from easycv.hooks import EvalHook
from easycv.hooks.best_ckpt_saver_hook import BestCkptSaverHook
from easycv.runner import EVRunner
from easycv.utils import get_root_logger


class DummyDataset(object):

    def __getitem__(self, idx):
        batch_size = 2
        output = {
            'img': torch.randn(batch_size, 3, 2, 2),
        }
        return output

    def __len__(self):
        return 2

    def evaluate(self, results, **kwargs):
        return {
            'ClsEvaluator_cifar10-1_neck_top1': 1.0,
            'ClsEvaluator_cifar10-2_neck_top1': 1.0
        }


def _build_model():

    class Model(nn.Module):

        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, img, mode='train'):
            if mode == 'test':
                return {'prob': [0.1, 0.2]}

            img = img.to('cpu')
            return self.linear(img)

        def train_step(self, img, optimizer, **kwargs):
            return dict(loss=self(img))

        def val_step(self, img, optimizer, **kwargs):
            return dict(loss=self(img))

    return Model()


class BestCkptSaverHookTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_best_ckpt_saver(self):
        work_dir = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        io.makedirs(work_dir)

        log_config = dict(
            interval=1,
            hooks=[
                dict(type='TextLoggerHook'),
                dict(type='TensorboardLoggerHook'),
            ])

        eval_cfg1 = dict(interval=1, mode='test', gpu_collect=True)
        evaluators1 = [
            dict(type='ClsEvaluator', topk=(1, ), dataset_name='cifar10-1')
        ]
        eval_cfg1['evaluators'] = build_evaluator(evaluators1)

        val_dataloader1 = DataLoader(DummyDataset())
        eval_hook1 = EvalHook(val_dataloader1, **eval_cfg1)

        eval_cfg2 = dict(interval=1, mode='test', gpu_collect=True)
        evaluators2 = [
            dict(type='ClsEvaluator', topk=(1, ), dataset_name='cifar10-2')
        ]
        eval_cfg2['evaluators'] = build_evaluator(evaluators2)
        val_dataloader2 = DataLoader(DummyDataset())
        eval_hook2 = EvalHook(val_dataloader2, **eval_cfg2)

        best_ckpt_hook = BestCkptSaverHook(
            by_epoch=True,
            save_optimizer=True,
            best_metric_name=[
                'ClsEvaluator_cifar10-1_neck_top1',
                'ClsEvaluator_cifar10-2_neck_top1'
            ],
            best_metric_type=['max', 'max'])

        model = _build_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.95)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(work_dir, '{}.log'.format(timestamp))
        logger = get_root_logger(log_file=log_file)
        runner = EVRunner(
            model=model, work_dir=work_dir, optimizer=optimizer, logger=logger)

        runner.register_logger_hooks(log_config)
        runner.register_checkpoint_hook(dict(interval=1))
        runner.register_hook(eval_hook1)
        runner.register_hook(eval_hook2)
        runner.register_hook(best_ckpt_hook)

        train_loader = DataLoader(torch.ones((5, 2)))
        runner.run([train_loader], [('train', 1)], 2)

        files = io.listdir(work_dir, recursive=True)
        self.assertIn('epoch_1.pth', files)
        self.assertIn('epoch_2.pth', files)
        self.assertIn('ClsEvaluator_cifar10-1_neck_top1_best.pth', files)
        self.assertIn('ClsEvaluator_cifar10-2_neck_top1_best.pth', files)
        self.assertTrue(io.exists(os.path.join(work_dir, 'tf_logs/')))
        io.rmtree(work_dir)


if __name__ == '__main__':
    unittest.main()
