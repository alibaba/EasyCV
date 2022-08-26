# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import os
import unittest

import numpy as np
from PIL import Image

from easycv.predictors.detector import TorchYoloXPredictor, DetrPredictor
from tests.ut_config import (PRETRAINED_MODEL_YOLOXS_EXPORT,
                             PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_JIT,
                             PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT,
                             DET_DATA_SMALL_COCO_LOCAL)
from numpy.testing import assert_array_almost_equal


class DetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_yolox_detector(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT

        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000522713.jpg')

        input_data_list = [np.asarray(Image.open(img))]
        predictor = TorchYoloXPredictor(
            model_path=detection_model_path, score_thresh=0.5)

        output = predictor.predict(input_data_list)[0]
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('detection_class_names', output)
        self.assertIn('ori_img_shape', output)

        self.assertEqual(len(output['detection_boxes']), 4)
        self.assertEqual(output['ori_img_shape'], [480, 640])

        self.assertListEqual(output['detection_classes'].tolist(),
                             np.array([13, 8, 8, 8], dtype=np.int32).tolist())

        self.assertListEqual(output['detection_class_names'],
                             ['bench', 'boat', 'boat', 'boat'])

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([0.92593855, 0.60268813, 0.57775956, 0.5750004],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[407.89523, 284.62598, 561.4984, 356.7296],
                      [439.37653, 263.42395, 467.01526, 271.79144],
                      [480.8597, 269.64435, 502.18765, 274.80127],
                      [510.37033, 268.4982, 527.67017, 273.04935]]),
            decimal=1)

    def test_yolox_detector_jit_nopre_notrt(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000522713.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        jit_path = PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_JIT
        predictor_jit = TorchYoloXPredictor(
            model_path=jit_path, score_thresh=0.5)

        output = predictor_jit.predict(input_data_list)[0]
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('detection_class_names', output)
        self.assertIn('ori_img_shape', output)

        self.assertEqual(len(output['detection_boxes']), 4)
        self.assertEqual(output['ori_img_shape'], [480, 640])

        self.assertListEqual(output['detection_classes'].tolist(),
                             np.array([13, 8, 8, 8], dtype=np.int32).tolist())

        self.assertListEqual(output['detection_class_names'],
                             ['bench', 'boat', 'boat', 'boat'])

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([0.92593855, 0.60268813, 0.57775956, 0.5750004],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[407.89523, 284.62598, 561.4984, 356.7296],
                      [439.37653, 263.42395, 467.01526, 271.79144],
                      [480.8597, 269.64435, 502.18765, 274.80127],
                      [510.37033, 268.4982, 527.67017, 273.04935]]),
            decimal=1)

    def test_yolox_detector_jit_pre_trt(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000522713.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        jit_path = PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_JIT
        predictor_jit = TorchYoloXPredictor(
            model_path=jit_path, score_thresh=0.5)

        output = predictor_jit.predict(input_data_list)[0]
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('detection_class_names', output)
        self.assertIn('ori_img_shape', output)

        self.assertEqual(len(output['detection_boxes']), 4)
        self.assertEqual(output['ori_img_shape'], [480, 640])

        self.assertListEqual(output['detection_classes'].tolist(),
                             np.array([13, 8, 8, 8], dtype=np.int32).tolist())

        self.assertListEqual(output['detection_class_names'],
                             ['bench', 'boat', 'boat', 'boat'])

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([0.92593855, 0.60268813, 0.57775956, 0.5750004],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
            np.array([[407.89523, 284.62598, 561.4984, 356.7296],
                      [439.37653, 263.42395, 467.01526, 271.79144],
                      [480.8597, 269.64435, 502.18765, 274.80127],
                      [510.37033, 268.4982, 527.67017, 273.04935]]),
            decimal=1)

    def test_vitdet_detector(self):
        model_path = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/detection/vitdet/vit_base/vitdet_maskrcnn_export.pth'
        img = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/demo/demo.jpg'
        out_file = './result.jpg'
        vitdet = DetrPredictor(model_path)
        output = vitdet.predict(img)
        vitdet.visualize(img, output, out_file=out_file)

        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('detection_masks', output)
        self.assertIn('img_metas', output)
        self.assertEqual(len(output['detection_boxes'][0]), 30)
        self.assertEqual(len(output['detection_scores'][0]), 30)
        self.assertEqual(len(output['detection_classes'][0]), 30)

        self.assertListEqual(
            output['detection_classes'][0].tolist(),
            np.array([
                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                2, 2, 2, 7, 7, 13, 13, 13, 56
            ],
                     dtype=np.int32).tolist())

        assert_array_almost_equal(
            output['detection_scores'][0],
            np.array([
                0.99791867, 0.99665856, 0.99480623, 0.99060905, 0.9882515,
                0.98319584, 0.9738879, 0.97290784, 0.9514897, 0.95104814,
                0.9321701, 0.86165, 0.8228847, 0.7623552, 0.76129806,
                0.6050861, 0.44348577, 0.3452973, 0.2895671, 0.22109479,
                0.21265312, 0.17855245, 0.1205352, 0.08981906, 0.10596471,
                0.05854294, 0.99749386, 0.9472857, 0.5945908, 0.09855112
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'][0],
            np.array([[294.7058, 117.29371, 378.83713, 149.99928],
                      [609.05444, 112.526474, 633.2971, 136.35175],
                      [481.4165, 110.987335, 522.5531, 130.01529],
                      [167.68184, 109.89049, 215.49057, 139.86987],
                      [374.75082, 110.68697, 433.10028, 136.23654],
                      [189.54971, 110.09322, 297.6167, 155.77412],
                      [266.5185, 105.37718, 326.54385, 127.916374],
                      [556.30225, 110.43166, 592.8248, 128.03764],
                      [432.49252, 105.086464, 484.0512, 132.272],
                      [0., 110.566444, 62.01249, 146.44017],
                      [591.74664, 110.43527, 619.73816, 126.68549],
                      [99.126854, 90.947975, 118.46699, 101.11096],
                      [59.895264, 94.110054, 85.60521, 106.67633],
                      [142.95819, 96.61966, 165.96964, 104.95929],
                      [83.062515, 89.802605, 99.1546, 98.69074],
                      [226.28802, 98.32568, 249.06772, 108.86408],
                      [136.67789, 94.75706, 154.62924, 104.289536],
                      [170.42459, 98.458694, 183.16309, 106.203156],
                      [67.56731, 89.68286, 82.62955, 98.35645],
                      [222.80092, 97.828445, 239.02655, 108.29377],
                      [134.34427, 92.31653, 149.19615, 102.97457],
                      [613.5186, 102.27066, 636.0434, 112.813644],
                      [607.4787, 110.87984, 630.1123, 127.65646],
                      [135.13664, 90.989876, 155.67192, 100.18036],
                      [431.61505, 105.43844, 484.36508, 132.50078],
                      [189.92722, 110.38832, 297.74353, 155.95557],
                      [220.67035, 177.13489, 455.32092, 380.45712],
                      [372.76584, 134.33807, 432.44357, 188.51534],
                      [50.403812, 110.543495, 70.4368, 119.65186],
                      [373.50272, 134.27258, 432.18475, 187.81824]]),
            decimal=1)


if __name__ == '__main__':
    unittest.main()
