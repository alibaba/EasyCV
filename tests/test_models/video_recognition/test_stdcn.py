# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch

from easycv.models.builder import build_model


class STGCNTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _get_model(self):
        model_cfg = dict(
            type='SkeletonGCN',
            backbone=dict(
                type='STGCN',
                in_channels=3,
                edge_importance_weighting=True,
                graph_cfg=dict(layout='coco', strategy='spatial')),
            cls_head=dict(
                type='STGCNHead',
                num_classes=60,
                in_channels=256,
                loss_cls=dict(type='CrossEntropyLoss')),
            train_cfg=None,
            test_cfg=None)
        model = build_model(model_cfg)
        return model

    def test_train(self):
        model = self._get_model()
        model.train()
        batch_size = 2
        keypoints = torch.randn([batch_size, 3, 300, 17, 2])
        label = torch.randint(0, 60, (batch_size, ))
        output = model(keypoint=keypoints, label=label)
        self.assertIn('loss_cls', output)
        self.assertIn('top1_acc', output)
        self.assertIn('top5_acc', output)

    def test_infer(self):
        model = self._get_model()
        model.eval()

        with torch.no_grad():
            keypoints = torch.randn([1, 3, 300, 17, 2])
            output = model(keypoint=keypoints, mode='test')
            self.assertEqual(output['prob'].shape, (1, 60))


if __name__ == '__main__':
    unittest.main()
