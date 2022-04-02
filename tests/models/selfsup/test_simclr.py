# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from torch import distributed as dist

from easycv.models.builder import build_model
from easycv.utils.test_util import pseudo_dist_init

_base_model_cfg = dict(
    type='SimCLR',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckSimCLR',  # SimCLR non-linear neck
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        num_layers=2,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1))


class SimCLRTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_simclr_train(self):
        model = build_model(_base_model_cfg).cuda()
        pseudo_dist_init()
        model.train()
        batch_size = 4
        imgs = [torch.randn(batch_size, 3, 224, 224).cuda()] * 2
        output = model(imgs, mode='train')

        self.assertIn('loss', output)
        self.assertEqual(output['loss'].shape, torch.Size([]))

        dist.destroy_process_group()

    def test_simclr_extract(self):
        model = build_model(_base_model_cfg).cuda()
        pseudo_dist_init()
        batch_size = 4
        imgs = torch.randn(batch_size, 3, 224, 224).cuda()
        output = model(imgs, mode='extract')
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape, torch.Size([4, 2048, 7, 7]))

        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
