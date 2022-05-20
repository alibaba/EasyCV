# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import os
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from easycv.predictors.detector import TorchYoloXPredictor
from tests.ut_config import (PRETRAINED_MODEL_YOLOXS_EXPORT,
                             PRETRAINED_MODEL_YOLOXS_EXPORT_JIT,
                             PRETRAINED_MODEL_YOLOXS_EXPORT_BLADE,
                             DET_DATA_SMALL_COCO_LOCAL)


class DetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_yolox_jit_detector(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT_JIT

        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        input_data_list = [np.asarray(Image.open(img))]
        predictor = TorchYoloXPredictor(
            model_path=detection_model_path, score_thresh=0.5)

        output = predictor.predict(input_data_list)[0]
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)

        self.assertEqual(len(output['detection_boxes']), 60)

    def test_yolox_blade_detector(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT_BLADE

        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        input_data_list = [np.asarray(Image.open(img))]
        predictor = TorchYoloXPredictor(
            model_path=detection_model_path, score_thresh=0.5)

        output = predictor.predict(input_data_list)[0]
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)

        self.assertEqual(len(output['detection_boxes']), 60)


if __name__ == '__main__':
    unittest.main()
