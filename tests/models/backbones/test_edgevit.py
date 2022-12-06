# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import EdgeVit


class EdgeVitTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_vitdet(self):
        model = EdgeVit(
            img_size=224,
            depth=[1, 1, 3, 2],
            embed_dim=[36, 72, 144, 288],
            head_dim=36,
            mlp_ratio=[4] * 4,
            qkv_bias=True,
            num_classes=1000,
            drop_path_rate=0.1,
            sr_ratios=[4, 2, 2, 1],
        )

        model.init_weights()
        model.train()
        imgs = torch.rand(36, 3, 224, 224)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([36, 288, 7, 7]))


if __name__ == '__main__':
    unittest.main()
