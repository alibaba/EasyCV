# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from easycv.predictors.detector import DetrPredictor


class FCOSTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_fcos(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/fcos_epoch_12.pth'
        config_path = 'configs/detection/fcos/fcos_r50_torch_1x_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        fcos = DetrPredictor(model_path, config_path)
        output = fcos.predict(img)
        fcos.visualize(img, output, out_file=None)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 100)
        self.assertEqual(len(output['detection_scores'][0]), 100)
        self.assertEqual(len(output['detection_classes'][0]), 100)

        self.assertListEqual(
            output['detection_classes'][0][:10].tolist(),
            np.array([0, 0, 0, 0, 0, 0, 0, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0][:10],
            np.array([
                0.16172607243061066, 0.13118137419223785, 0.12351018935441971,
                0.11615370959043503, 0.09833250194787979, 0.0773085504770279,
                0.07507805526256561, 0.7142091989517212, 0.6164696216583252,
                0.5857587456703186
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0][:10],
            np.array([[
                255.08067321777344, 102.54728698730469, 261.5584411621094,
                112.76062774658203
            ],
                      [
                          375.2182312011719, 119.94615173339844,
                          381.58447265625, 133.05909729003906
                      ],
                      [
                          360.9225769042969, 108.36721801757812,
                          368.409423828125, 120.57501220703125
                      ],
                      [
                          241.30831909179688, 100.16476440429688,
                          249.76853942871094, 108.0853500366211
                      ],
                      [
                          263.5992736816406, 97.13397216796875,
                          270.6929626464844, 112.32050323486328
                      ],
                      [
                          234.89877319335938, 98.97943115234375,
                          249.2810821533203, 108.02184295654297
                      ],
                      [
                          371.852294921875, 134.10707092285156,
                          432.510986328125, 187.67025756835938
                      ],
                      [
                          294.9649353027344, 116.47904968261719,
                          378.7293701171875, 149.90737915039062
                      ],
                      [
                          480.3441467285156, 110.31671142578125,
                          523.027099609375, 130.33409118652344
                      ],
                      [
                          398.2224426269531, 110.64815521240234,
                          433.01568603515625, 133.15269470214844
                      ]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
