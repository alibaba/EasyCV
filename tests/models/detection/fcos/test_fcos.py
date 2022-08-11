# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal
from easycv.predictors.detector import DetectionPredictor

class FCOSTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_fcos(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/fcos/epoch_12.pth'
        config_path = 'configs/detection/fcos/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_1x_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        fcos = DetectionPredictor(model_path, config_path)
        output = fcos.predict(img)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 10)
        self.assertEqual(len(output['detection_scores'][0]), 10)
        self.assertEqual(len(output['detection_classes'][0]), 10)

        print(output['detection_boxes'][0].tolist())
        print(output['detection_scores'][0].tolist())
        print(output['detection_classes'][0].tolist())

        self.assertListEqual(
            output['detection_classes'][0].tolist(),
            np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 13], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0],
            np.array([
                0.6641181707382202, 0.6135501265525818, 0.5985610485076904,
                0.5694775581359863, 0.5586040616035461, 0.5209507942199707,
                0.5056729912757874, 0.4943872094154358, 0.4850597083568573,
                0.45443734526634216
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0],
            np.array([[
                295.5196228027344, 116.56035614013672, 380.0883483886719,
                150.24908447265625
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
                      ],
                      [
                          98.66332244873047, 89.88417053222656,
                          118.9398422241211, 101.25397491455078
                      ],
                      [
                          167.9045867919922, 109.57560729980469,
                          209.74375915527344, 139.98898315429688
                      ],
                      [
                          591.0496826171875, 110.55867767333984,
                          619.4395751953125, 126.65755462646484
                      ],
                      [
                          218.92051696777344, 177.0509033203125,
                          455.8321838378906, 385.0356140136719
                      ]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
