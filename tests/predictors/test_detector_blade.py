# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import os
import unittest
import numpy as np
from PIL import Image
from easycv.predictors.detector import TorchYoloXPredictor
from tests.ut_config import (PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_BLADE,
                             PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_BLADE,
                             DET_DATA_SMALL_COCO_LOCAL)

import torch
from numpy.testing import assert_array_almost_equal


@unittest.skipIf(torch.__version__ != '1.8.1+cu102',
                 'Blade need another environment')
class DetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_yolox_detector_blade_nopre_notrt(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000522713.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        blade_path = PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_BLADE
        predictor_blade = TorchYoloXPredictor(
            model_path=blade_path, score_thresh=0.5)

        output = predictor_blade.predict(input_data_list)[0]
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

    def test_yolox_detector_blade_pre_notrt(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000522713.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        blade_path = PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_BLADE
        predictor_blade = TorchYoloXPredictor(
            model_path=blade_path, score_thresh=0.5)

        output = predictor_blade.predict(input_data_list)[0]
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


if __name__ == '__main__':
    unittest.main()
