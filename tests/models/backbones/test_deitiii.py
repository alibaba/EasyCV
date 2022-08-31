# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal

from easycv.predictors.classifier import ClsPredictor


class DeiTIIITest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_deitiii(self):
        model_path = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/deitiii/epoch_800.pth'
        config_path = 'configs/classification/imagenet/vit/imagenet_deitiii_large_patch16_192_jpg.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/deitiii_demo.JPEG'
        deitiii = ClsPredictor(model_path, config_path)
        output = deitiii.predict(img)

        self.assertIn('prob', output)
        self.assertIn('class', output)
        self.assertEqual(len(output['prob'][0]), 1000)

        self.assertListEqual(
            output['prob'][0][:10].numpy().tolist(),
            torch.Tensor([
                2.04629918698628899e-06, 5.27398606209317222e-06,
                5.52915162188583054e-06, 3.60625563189387321e-06,
                3.29447357216849923e-06, 5.61309570912271738e-06,
                8.93703327164985240e-06, 4.89157764604897238e-06,
                4.39371024185675196e-06, 5.21611764270346612e-06
            ]).numpy().tolist())


if __name__ == '__main__':
    unittest.main()
