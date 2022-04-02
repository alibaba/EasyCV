# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import LiteHRNet


class LiteHRNetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _test_litehrnet(self, module_type):
        extra = dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=(module_type, module_type, module_type),
                with_fuse=(True, True, True),
                reduce_ratios=(8, 8, 8),
                num_channels=((40, 80), (40, 80, 160), (40, 80, 160, 320))),
            with_head=True)

        model = LiteHRNet(extra, in_channels=3)
        model.init_weights()
        model.train()
        imgs = torch.randn(2, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([2, 40, 56, 56]))

    def test_lite(self):
        self._test_litehrnet(module_type='LITE')

    def test_naive(self):
        self._test_litehrnet(module_type='NAIVE')


if __name__ == '__main__':
    unittest.main()
