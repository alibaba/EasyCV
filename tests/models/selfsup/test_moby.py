# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from torch import distributed as dist

from easycv.models.builder import build_model
from easycv.utils.test_util import pseudo_dist_init

_base_model_cfg = dict(
    type='MoBY',
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    queue_len=4096,
    momentum=0.99,
    pretrained=None,
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='resnet50',  # 2048
        num_classes=0,
        pretrained=True,
    ),
    neck=dict(
        type='MoBYMLP',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        num_layers=2),
    head=dict(
        type='MoBYMLP',
        in_channels=256,
        hid_channels=4096,
        out_channels=256,
        num_layers=2))


class MoBYTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_moby_train(self):
        model = build_model(_base_model_cfg).cuda()
        pseudo_dist_init()
        model.train()
        batch_size = 4
        imgs = [torch.randn(batch_size, 3, 224, 224).cuda()] * 2
        output = model(imgs, mode='train')

        self.assertIn('loss', output)
        self.assertEqual(output['loss'].shape, torch.Size([]))
        dist.destroy_process_group()

    def test_moby_extract(self):
        model = build_model(_base_model_cfg).cuda()
        pseudo_dist_init()
        batch_size = 4
        imgs = torch.randn(batch_size, 3, 224, 224).cuda()
        output = model(imgs, mode='extract')
        self.assertEqual(output['neck'].shape, torch.Size([4, 2048, 1, 1]))
        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
