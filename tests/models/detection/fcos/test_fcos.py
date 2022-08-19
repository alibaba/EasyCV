# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from easycv.predictors.detector import DetectorPredictor


class FCOSTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_fcos(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/epoch_12.pth'
        config_path = 'configs/detection/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        fcos = DetectorPredictor(model_path, config_path)
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
            np.array([0, 0, 0, 0, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0][:10],
            np.array([
                0.12291150540113449, 0.11793176084756851, 0.09635727107524872,
                0.07252732664346695, 0.6641181707382202, 0.6135501265525818,
                0.5985610485076904, 0.5694775581359863, 0.5586040616035461,
                0.5209507942199707
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0][:10],
            np.array([[
                236.57887268066406, 99.18157196044922, 250.55189514160156,
                109.49268341064453
            ],
                      [
                          375.9055480957031, 120.44611358642578,
                          381.8572692871094, 133.46702575683594
                      ],
                      [
                          225.96961975097656, 98.12339782714844,
                          250.6289825439453, 109.06866455078125
                      ],
                      [
                          532.4306030273438, 109.92780303955078,
                          540.611572265625, 125.47993469238281
                      ],
                      [
                          295.5196228027344, 116.56035614013672,
                          380.0883483886719, 150.24908447265625
                      ],
                      [
                          190.57131958007812, 108.96343231201172,
                          297.7738037109375, 154.69515991210938
                      ],
                      [
                          480.5726013183594, 110.4341812133789,
                          522.8551635742188, 129.9452667236328
                      ],
                      [
                          431.1232604980469, 105.17676544189453,
                          483.89617919921875, 131.85870361328125
                      ],
                      [
                          398.6544494628906, 110.90837860107422,
                          432.6370849609375, 132.89173889160156
                      ],
                      [
                          609.3126831054688, 111.62432861328125,
                          635.4577026367188, 137.03529357910156
                      ]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
