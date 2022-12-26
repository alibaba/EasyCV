# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import numpy as np
import torch

from easycv.models import Recognizer3D
from easycv.models.builder import build_model


class Recognizer3dTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        model_cfg = dict(
            type='Recognizer3D',
            backbone=dict(
                type='SwinTransformer3D',
                patch_size=(4, 4, 4),
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=(8, 7, 7),
                mlp_ratio=4.,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.,
                attn_drop_rate=0.,
                drop_path_rate=0.2,
                patch_norm=True),
            cls_head=dict(
                type='I3DHead',
                in_channels=768,
                num_classes=400,
                spatial_type='avg',
                dropout_ratio=0.5),
            test_cfg=dict(average_clips='prob'))

        model = build_model(model_cfg)

        model.train()
        # batch, num_clip, channel, time, w, h
        frames = torch.randn(1, 1, 3, 32, 224, 224)
        label = torch.tensor([1])
        output = model(imgs=frames, label=label)

        self.assertIn('loss_cls', output)

        model.eval()
        with torch.no_grad():
            frames_test = torch.randn(1, 12, 3, 32, 224, 224)
            output = model(imgs=frames_test, mode='test')
            self.assertEqual(output['prob'].shape, (1, 400))

    def test_x3d(self):
        model_cfg = dict(
            type='Recognizer3D',
            backbone=dict(
                type='X3D',
                width_factor=2.0,
                depth_factor=2.2,
                bottlneck_factor=2.25,
                dim_c5=2048,
                dim_c1=12,
                num_classes=400,
                num_frames=4,
            ),
            cls_head=dict(
                type='X3DHead',
                in_channels=192,
                num_classes=400,
                spatial_type='avg',
                dropout_ratio=0.5,
                fc1_bias=False),
            test_cfg=dict(average_clips='prob'))

        model = build_model(model_cfg)

        model.train()
        # batch, num_clip, channel, time, w, h
        frames = torch.randn(1, 1, 3, 4, 224, 224)
        label = torch.tensor([1])
        output = model(imgs=frames, label=label)

        self.assertIn('loss_cls', output)

        model.eval()
        with torch.no_grad():
            frames_test = torch.randn(1, 12, 3, 4, 224, 224)
            output = model(imgs=frames_test, mode='test')
            self.assertEqual(output['prob'].shape, (1, 400))


if __name__ == '__main__':
    unittest.main()
