# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones import ViTDet


class ViTDetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_vitdet(self):
        model = ViTDet(
            img_size=1024,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            drop_path_rate=0.1,
            window_size=14,
            mlp_ratio=4,
            qkv_bias=True,
            window_block_indexes=[
                # 2, 5, 8 11 for global attention
                0,
                1,
                3,
                4,
                6,
                7,
                9,
                10,
            ],
            residual_block_indexes=[],
            use_rel_pos=True)

        model.init_weights()
        model.train()
        imgs = torch.randn(2, 3, 1024, 1024)
        feat = model(imgs)
        self.assertEqual(len(feat), 1)
        self.assertEqual(feat[0].shape, torch.Size([2, 768, 64, 64]))
