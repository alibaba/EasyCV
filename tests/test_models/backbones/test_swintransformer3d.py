# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import SwinTransformer3D


class SwinTransformer3dTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        model = SwinTransformer3D()
        model.train()
        imgs = torch.randn(1, 3, 32, 224, 224)
        feat = model(imgs)
        self.assertEqual(feat.shape, torch.Size([1, 768, 8, 7, 7]))


if __name__ == '__main__':
    unittest.main()
