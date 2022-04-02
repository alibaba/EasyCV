# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.builder import build_model

_base_model_cfg = dict(
    type='MAE',
    backbone=dict(
        type='MaskedAutoencoderViT',
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
    ),
    neck=dict(
        type='MAENeck',
        embed_dim=768,
        patch_size=16,
        in_chans=3,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.,
    ),
    mask_ratio=0.75,
    norm_pix_loss=True)


class MAETest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_mae_train(self):
        model = build_model(_base_model_cfg)
        model.train()

        batch_size = 2
        imgs = torch.randn(batch_size, 3, 224, 224)

        output = model(imgs, mode='train')
        self.assertEqual(output['loss'].shape, torch.Size([]))


if __name__ == '__main__':
    unittest.main()
