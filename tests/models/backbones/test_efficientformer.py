# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import EfficientFormer


class EfficientFormerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_vitdet(self):
        model = EfficientFormer(
            layers=[3, 2, 6, 4],
            embed_dims=[48, 96, 224, 448],
            downsamples=[True, True, True, True],
            vit_num=1,
            fork_feat=True,
            distillation=True,
        )

        model.init_weights()
        model.train()
        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 4)
        self.assertEqual(feat[0].shape, torch.Size([2, 56, 56, 48]))
        self.assertEqual(feat[1].shape, torch.Size([2, 28, 28, 96]))
        self.assertEqual(feat[2].shape, torch.Size([2, 14, 14, 224]))
        self.assertEqual(feat[3].shape, torch.Size([2, 49, 448]))
