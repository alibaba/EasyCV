# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from torch import distributed as dist

from easycv.models.builder import build_model
from easycv.utils.test_util import pseudo_dist_init

_num_crops = [2, 6]
_base_model_cfg = model = dict(
    type='SWAV',
    pretrained=None,
    train_preprocess=['randomGrayScale', 'gaussianBlur'],
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    # swav need  mulit crop ,doesn't support vit based model
    neck=dict(
        type='NonLinearNeckSwav',
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=False),
    config=dict(
        # multi crop setting
        num_crops=_num_crops,
        size_crops=[160, 96],
        min_scale_crops=[0.14, 0.05],
        max_scale_crops=[1, 0.14],

        # swav setting
        crops_for_assign=[0, 1],
        epsilon=0.05,
        nmb_prototypes=3000,
        sinkhorn_iterations=3,
        temperature=0.1,

        # queue setting
        queue_length=3840,
        epoch_queue_starts=15))


class SWAVTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_swav_train(self):
        model = build_model(_base_model_cfg).cuda()
        pseudo_dist_init()
        model.train()
        batch_size = 4
        imgs = [torch.randn(batch_size, 3, 224, 224).cuda()] * 8
        output = model(imgs, mode='train')

        self.assertIn('loss', output)
        self.assertEqual(output['loss'].shape, torch.Size([]))

        dist.destroy_process_group()

    def test_swav_extract(self):
        model = build_model(_base_model_cfg).cuda()
        pseudo_dist_init()
        batch_size = 4
        imgs = torch.randn(batch_size, 3, 224, 224).cuda()
        output = model(imgs, mode='extract')
        self.assertEqual(output['neck'].shape, torch.Size([4, 128]))

        dist.destroy_process_group()


if __name__ == '__main__':
    unittest.main()
