# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import ResNeXt


class ResNeXtTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_resnext(self):
        model = ResNeXt(
            depth=50,
            groups=32,
            base_width=4,
            out_indices=[4],
            norm_cfg=dict(type='BN'))
        model.init_weights()
        model.train()
        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([2, 2048, 7, 7]))
