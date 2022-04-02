# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import HRNet


class HRNetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_hrnet(self):
        extra = dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(32, 64)),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(32, 64, 128)),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(32, 64, 128, 256)))

        model = HRNet(extra=extra, in_channels=3)
        model.init_weights()
        model.train()
        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([2, 32, 56, 56]))


if __name__ == '__main__':
    unittest.main()
