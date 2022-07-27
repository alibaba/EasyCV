# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.builder import build_head


class UperHeadTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_forward_train(self):
        norm_cfg = dict(type='BN', requires_grad=True)
        uper_head_config = dict(
            type='UPerHead',
            in_channels=[256, 512, 1024, 2048],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        head = build_head(uper_head_config)
        head = head.to('cuda')

        batch_size = 2
        dummy_inputs = [
            torch.rand(batch_size, 256, 128, 128).to('cuda'),
            torch.rand(batch_size, 512, 64, 64).to('cuda'),
            torch.rand(batch_size, 1024, 32, 32).to('cuda'),
            torch.rand(batch_size, 2048, 16, 16).to('cuda'),
        ]

        gt_semantic_seg = torch.randint(
            low=0, high=19, size=(batch_size, 1, 512, 512)).to('cuda')
        train_output = head.forward_train(
            dummy_inputs,
            img_metas=None,
            gt_semantic_seg=gt_semantic_seg,
            train_cfg=None)
        self.assertIn('loss_ce', train_output)
        self.assertIn('acc_seg', train_output)
        self.assertEqual(train_output['acc_seg'].shape, torch.Size([1]))

    def test_forward_test(self):
        norm_cfg = dict(type='BN', requires_grad=True)
        uper_head_config = dict(
            type='UPerHead',
            in_channels=[256, 512, 1024, 2048],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))

        head = build_head(uper_head_config)
        head = head.to('cuda')

        batch_size = 2
        dummy_inputs = [
            torch.rand(batch_size, 256, 128, 128).to('cuda'),
            torch.rand(batch_size, 512, 64, 64).to('cuda'),
            torch.rand(batch_size, 1024, 32, 32).to('cuda'),
            torch.rand(batch_size, 2048, 16, 16).to('cuda'),
        ]

        with torch.no_grad():
            test_output = head.forward_test(
                dummy_inputs, img_metas=None, test_cfg=None)
        self.assertEqual(test_output.shape, torch.Size([2, 19, 128, 128]))


if __name__ == '__main__':
    unittest.main()
