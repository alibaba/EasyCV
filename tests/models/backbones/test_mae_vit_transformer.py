# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.backbones.mae_vit_transformer import MaskedAutoencoderViT


class MaskedAutoencoderViTTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_masked_auto_encoder_vit(self):
        model = MaskedAutoencoderViT()
        model.train()
        imgs = torch.randn(2, 3, 224, 224)
        output = model(imgs, mask_ratio=0.75)
        self.assertEqual(output[0].shape, torch.Size([2, 50, 1024]))
        self.assertEqual(output[1].shape, torch.Size([2, 196]))
        self.assertEqual(output[2].shape, torch.Size([2, 196]))


if __name__ == '__main__':
    unittest.main()
