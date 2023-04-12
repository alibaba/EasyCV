# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import X3D


class X3DTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        model = X3D()
        model.train()
        # b, channel, time, w, h
        imgs = torch.randn(1, 3, 32, 224, 224)
        feat = model(imgs)
        self.assertEqual(feat.shape, torch.Size([1, 192, 32, 7, 7]))


if __name__ == '__main__':
    unittest.main()
