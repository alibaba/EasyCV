# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal

from easycv.predictors.detector import DetrPredictor


class DETRTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/detr/epoch_150.pth'
        config_path = 'configs/detection/detr/detr_r50_8x2_150e_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        detr = DetrPredictor(model_path, config_path)
        output = detr.predict(img)
        detr.visualize(img, output, out_file=None)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 100)
        self.assertEqual(len(output['detection_scores'][0]), 100)
        self.assertEqual(len(output['detection_classes'][0]), 100)

        self.assertListEqual(
            output['detection_classes'][0][:10].tolist(),
            np.array([2, 0, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0][:10],
            np.array([
                0.07836595922708511, 0.219977006316185, 0.5831383466720581,
                0.4256463646888733, 0.9853266477584839, 0.24607707560062408,
                0.28005731105804443, 0.500579833984375, 0.09835881739854813,
                0.05178987979888916
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0][:10],
            np.array([[
                131.10389709472656, 90.93302154541016, 148.95504760742188,
                101.69216918945312
            ],
                      [
                          239.10910034179688, 113.36551666259766,
                          256.0523376464844, 125.22894287109375
                      ],
                      [
                          132.1316375732422, 90.8366470336914,
                          151.00839233398438, 101.83119201660156
                      ],
                      [
                          579.37646484375, 108.26667785644531,
                          605.0717163085938, 124.79525756835938
                      ],
                      [
                          189.69073486328125, 108.04875946044922,
                          296.8011779785156, 154.44204711914062
                      ],
                      [
                          588.5413208007812, 107.89535522460938,
                          615.6463012695312, 124.41362762451172
                      ],
                      [
                          57.38536071777344, 89.7335433959961,
                          79.20274353027344, 102.61941528320312
                      ],
                      [
                          163.97628784179688, 92.95049285888672,
                          180.87033081054688, 102.6163330078125
                      ],
                      [
                          127.82454681396484, 90.27918243408203,
                          144.6781768798828, 99.71304321289062
                      ],
                      [
                          438.4545593261719, 103.00477600097656,
                          480.4275817871094, 121.69993591308594
                      ]]),
            decimal=1)

    def test_dab_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dab_detr/dab_detr_epoch_50.pth'
        config_path = 'configs/detection/dab_detr/dab_detr_r50_8x2_50e_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        dab_detr = DetrPredictor(model_path, config_path)
        output = dab_detr.predict(img)
        dab_detr.visualize(img, output, out_file=None)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 300)
        self.assertEqual(len(output['detection_scores'][0]), 300)
        self.assertEqual(len(output['detection_classes'][0]), 300)

        self.assertListEqual(
            output['detection_classes'][0][:10].tolist(),
            np.array([2, 2, 13, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0][:10],
            np.array([
                0.7688284516334534, 0.7646799683570862, 0.7159939408302307,
                0.6902833580970764, 0.6633996367454529, 0.6523147821426392,
                0.633848249912262, 0.6229104995727539, 0.611840009689331,
                0.5631589293479919
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0][:10],
            np.array([[
                294.2984313964844, 116.07160949707031, 380.4406433105469,
                149.6365509033203
            ],
                      [
                          480.0610656738281, 109.99347686767578,
                          523.2314453125, 130.26318359375
                      ],
                      [
                          220.32269287109375, 176.51010131835938,
                          456.51715087890625, 386.30767822265625
                      ],
                      [
                          167.6925506591797, 108.25935363769531,
                          214.93780517578125, 138.94424438476562
                      ],
                      [
                          398.1152648925781, 111.34457397460938,
                          433.72052001953125, 133.36280822753906
                      ],
                      [
                          430.48736572265625, 104.4018325805664,
                          484.1470947265625, 132.18893432617188
                      ],
                      [
                          607.396728515625, 111.72560119628906,
                          637.2987670898438, 136.2375946044922
                      ],
                      [
                          267.43353271484375, 105.93965911865234,
                          327.1937561035156, 130.18527221679688
                      ],
                      [
                          589.790771484375, 110.36975860595703,
                          618.8001098632812, 126.1950454711914
                      ],
                      [
                          0.3374290466308594, 110.91182708740234,
                          63.00359344482422, 146.0926971435547
                      ]]),
            decimal=1)

    def test_dn_detr(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/dn_detr/dn_detr_epoch_50.pth'
        config_path = 'configs/detection/dab_detr/dn_detr_r50_8x2_50e_coco.py'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        dn_detr = DetrPredictor(model_path, config_path)
        output = dn_detr.predict(img)
        dn_detr.visualize(img, output, out_file=None)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 300)
        self.assertEqual(len(output['detection_scores'][0]), 300)
        self.assertEqual(len(output['detection_classes'][0]), 300)

        self.assertListEqual(
            output['detection_classes'][0][:10].tolist(),
            np.array([2, 13, 2, 2, 2, 2, 2, 2, 2, 2], dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0][:10],
            np.array([
                0.8800525665283203, 0.866659939289093, 0.8665854930877686,
                0.8030595183372498, 0.7642921209335327, 0.7375038862228394,
                0.7270554304122925, 0.6710091233253479, 0.6316548585891724,
                0.6164721846580505
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0][:10],
            np.array([[
                294.9338073730469, 115.7542495727539, 377.5517578125,
                150.59274291992188
            ],
                      [
                          220.57424926757812, 175.97023010253906,
                          456.9001770019531, 383.2597351074219
                      ],
                      [
                          479.5928649902344, 109.94012451171875,
                          523.7343139648438, 130.80604553222656
                      ],
                      [
                          398.6956787109375, 111.45973205566406,
                          434.0437316894531, 134.1909637451172
                      ],
                      [
                          166.98208618164062, 109.44792938232422,
                          210.35342407226562, 139.9746856689453
                      ],
                      [
                          609.432373046875, 113.08062744140625,
                          635.9082641601562, 136.74383544921875
                      ],
                      [
                          268.0716552734375, 105.00788879394531,
                          327.4037170410156, 128.01449584960938
                      ],
                      [
                          190.77467346191406, 107.42850494384766,
                          298.35760498046875, 156.2850341796875
                      ],
                      [
                          591.0296020507812, 110.53913116455078,
                          620.702880859375, 127.42123413085938
                      ],
                      [
                          431.6607971191406, 105.04813385009766,
                          484.4869689941406, 132.45864868164062
                      ]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
