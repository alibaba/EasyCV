# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
import torch

from easycv.predictors.classifier import ClassificationPredictor


class HydraAttentionTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_hydraAttention(self):
        model_path = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/hydra_attention/deit_base_patch16_224%20(Hydra%20Attention%20%5B12%20layers%5D).pth'
        config_path = 'configs/classification/imagenet/vit/deit_base_hydraAtt_patch16_224.py'

        img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pipelines = [
            dict(type='ToPILImage'),
            dict(type='Resize', size=256, interpolation=3),
            dict(type='CenterCrop', size=224),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Collect', keys=['img'])
        ]

        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        img = np.array([[[np.random.random() for k in range(3)]
                         for j in range(256)] for i in range(256)])
        img = np.uint8(img * 255)

        hydraAttention = ClassificationPredictor(
            model_path,
            config_path,
            label_map_path=None,
            pil_input=False,
            pipelines=pipelines)

        output = hydraAttention([img])[0]

        self.assertIn('class', output)
        self.assertEqual(len(output['class']), 1)
        self.assertEqual(int(output['class'][0]), 284)


if __name__ == '__main__':
    unittest.main()
