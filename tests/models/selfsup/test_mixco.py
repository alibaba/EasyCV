# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from torch import distributed as dist

from easycv.models.builder import build_model
from easycv.utils.test_util import pseudo_dist_init

_base_model_cfg = dict(
    type='MIXCO',
    pretrained=None,
    train_preprocess=['randomGrayScale', 'gaussianBlur', 'mixUp'],
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV1',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.2),
    mixco_head=dict(type='ContrastiveHead', temperature=0.05),
)


class MIXCOTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_mixco_train(self):
        model = build_model(_base_model_cfg).cuda()
        pseudo_dist_init()
        model.train()
        batch_size = 4
        imgs = [torch.randn(batch_size, 3, 224, 224).cuda()] * 2
        output = model(imgs, mode='train')

        self.assertIn('loss', output)
        self.assertEqual(output['loss'].shape, torch.Size([]))

        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
