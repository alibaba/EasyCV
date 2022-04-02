# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.builder import build_model

_base_model_cfg = dict(
    type='BYOL',
    pretrained=None,
    base_momentum=0.996,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV2',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        size_average=True,
        predictor=dict(
            type='NonLinearNeckV2',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            with_avg_pool=False)))


class BYOLTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_byol_train(self):
        model = build_model(_base_model_cfg)
        model.train()
        model.init_weights()

        batch_size = 3
        imgs = [torch.randn(batch_size, 3, 640, 640)] * 2
        output = model(imgs, mode='train')
        self.assertIn('loss', output)

    def test_byol_extract(self):
        model = build_model(_base_model_cfg)

        batch_size = 3
        imgs = torch.randn(batch_size, 3, 640, 640)
        output = model(imgs, mode='extract')
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0].shape, torch.Size([3, 2048, 20, 20]))


if __name__ == '__main__':
    unittest.main()
