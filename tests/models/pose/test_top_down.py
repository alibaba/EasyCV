# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch

from easycv.models import TopDown


def gen_fake_data(input_shape, num_joints=17):
    (N, C, H, W) = input_shape

    target = np.zeros([N, num_joints, H // 4, W // 4], dtype=np.float32)
    target_weight = np.ones([N, num_joints, 1], dtype=np.float32)

    img_metas = [{
        'img_shape': (H, W, C),
        'center': np.array([W / 2, H / 2]),
        'scale': np.array([0.5, 0.5]),
        'flip_pairs': [],
        'image_file': 'demo.jpg',
        'image_id': 1,
        'inference_channel': np.arange(num_joints)
    } for _ in range(N)]

    fake_inputs = {
        'target': torch.FloatTensor(target),
        'target_weight': torch.FloatTensor(target_weight),
        'img_metas': img_metas
    }
    fake_inputs['imgs'] = torch.rand(input_shape).requires_grad_(True)

    return fake_inputs


class TopDownTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _test_topdown(self, input_cfg, model_cfg):
        detector = TopDown(model_cfg['backbone'], None,
                           model_cfg['keypoint_head'], model_cfg['train_cfg'],
                           model_cfg['test_cfg'], model_cfg['pretrained'])

        detector.init_weights()

        imgs = input_cfg.pop('imgs')
        target = input_cfg.pop('target')
        target_weight = input_cfg.pop('target_weight')
        img_metas = input_cfg.pop('img_metas')

        # Test forward train
        train_output = detector.forward(
            img=imgs,
            target=target,
            target_weight=target_weight,
            img_metas=img_metas)

        # Test forward test
        with torch.no_grad():
            test_output = detector.forward(
                img=imgs, mode='test', img_metas=img_metas)

        return train_output, test_output

    def test_litehrnet30(self):
        model_cfg = dict(
            type='TopDown',
            pretrained=None,
            backbone=dict(
                type='LiteHRNet',
                in_channels=3,
                extra=dict(
                    stem=dict(
                        stem_channels=32, out_channels=32, expand_ratio=1),
                    num_stages=3,
                    stages_spec=dict(
                        num_modules=(3, 8, 3),
                        num_branches=(2, 3, 4),
                        num_blocks=(2, 2, 2),
                        module_type=('LITE', 'LITE', 'LITE'),
                        with_fuse=(True, True, True),
                        reduce_ratios=(8, 8, 8),
                        num_channels=(
                            (40, 80),
                            (40, 80, 160),
                            (40, 80, 160, 320),
                        )),
                    with_head=True,
                )),
            keypoint_head=dict(
                type='TopdownHeatmapSimpleHead',
                in_channels=40,
                out_channels=17,
                num_deconv_layers=0,
                extra=dict(final_conv_kernel=1, ),
                loss_keypoint=dict(
                    type='JointsMSELoss', use_target_weight=True)),
            train_cfg=dict(),
            test_cfg=dict(
                flip_test=True,
                post_process='default',
                shift_heatmap=True,
                modulate_kernel=11))

        fake_inputs = gen_fake_data(input_shape=(1, 3, 256, 256))

        train_output, test_output = self._test_topdown(fake_inputs, model_cfg)

        self.assertTrue(isinstance(train_output, dict))
        self.assertIn('mse_loss', train_output)

        self.assertTrue(isinstance(test_output, dict))
        self.assertIn('preds', test_output)


if __name__ == '__main__':
    unittest.main()
