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

    def _assert_results(self, results):
        self.assertEqual(results['ori_img_shape'], [480, 640])
        self.assertListEqual(results['detection_classes'].tolist(),
                             np.array([13, 8, 8, 8], dtype=np.int32).tolist())
        self.assertListEqual(results['detection_class_names'],
                             ['bench', 'boat', 'boat', 'boat'])
        assert_array_almost_equal(
            results['detection_scores'],
            np.array([0.92335737, 0.59416807, 0.5567955, 0.55368793],
                     dtype=np.float32),
            decimal=2)
        assert_array_almost_equal(
            results['detection_boxes'],
            np.array([[408.1708, 285.11456, 561.84924, 356.42285],
                      [438.88098, 264.46606, 467.07275, 271.76355],
                      [510.19467, 268.46664, 528.26935, 273.37192],
                      [480.9472, 269.74115, 502.00842, 274.85553]]),
            decimal=1)

    def test_yolox_detector_blade_nopre_notrt(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000522713.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        blade_path = PRETRAINED_MODEL_YOLOXS_NOPRE_NOTRT_BLADE
        predictor_blade = TorchYoloXPredictor(
            model_path=blade_path, score_thresh=0.5)

        output = predictor_blade.predict(input_data_list)[0]
        self._assert_results(output)

    def test_yolox_detector_blade_pre_notrt(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000522713.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        blade_path = PRETRAINED_MODEL_YOLOXS_PRE_NOTRT_BLADE
        predictor_blade = TorchYoloXPredictor(
            model_path=blade_path, score_thresh=0.5)

        output = predictor_blade.predict(input_data_list)[0]
        self._assert_results(output)


if __name__ == '__main__':
    unittest.main()
