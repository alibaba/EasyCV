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
                             PRETRAINED_MODEL_YOLOXS_END2END_JIT,
                             PRETRAINED_MODEL_YOLOXS_END2END_BLADE,
                             DET_DATA_SMALL_COCO_LOCAL)

from tests.toolkit.time_cost import benchmark
import logging
import pandas as pd
import torch
from numpy.testing import assert_array_almost_equal


@unittest.skipIf(torch.__version__ == '1.8.0',
                 'Blade need another environment')
class DetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_end2end(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        jit_path = PRETRAINED_MODEL_YOLOXS_END2END_JIT
        blade_path = PRETRAINED_MODEL_YOLOXS_END2END_BLADE

        predictor_jit = TorchYoloXPredictor(
            model_path=jit_path, score_thresh=0.5)

        predictor_blade = TorchYoloXPredictor(
            model_path=blade_path, score_thresh=0.5)

        output_jit = predictor_jit.predict(input_data_list)[0]
        output_blade = predictor_blade.predict(input_data_list)[0]

        self.assertIn('detection_boxes', output_jit)
        self.assertIn('detection_scores', output_jit)
        self.assertIn('detection_classes', output_jit)

        self.assertIn('detection_boxes', output_blade)
        self.assertIn('detection_scores', output_blade)
        self.assertIn('detection_classes', output_blade)

        assert_array_almost_equal(
            output_jit['detection_boxes'],
            output_blade['detection_boxes'],
            decimal=3)
        assert_array_almost_equal(
            output_jit['detection_classes'],
            output_blade['detection_classes'],
            decimal=3)
        assert_array_almost_equal(
            output_jit['detection_scores'],
            output_blade['detection_scores'],
            decimal=3)

    def test_export(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        jit_path = PRETRAINED_MODEL_YOLOXS_EXPORT_JIT
        blade_path = PRETRAINED_MODEL_YOLOXS_EXPORT_BLADE

        predictor_jit = TorchYoloXPredictor(
            model_path=jit_path, score_thresh=0.5)

        predictor_blade = TorchYoloXPredictor(
            model_path=blade_path, score_thresh=0.5)

        output_jit = predictor_jit.predict(input_data_list)[0]
        output_blade = predictor_blade.predict(input_data_list)[0]

        self.assertIn('detection_boxes', output_jit)
        self.assertIn('detection_scores', output_jit)
        self.assertIn('detection_classes', output_jit)

        self.assertIn('detection_boxes', output_blade)
        self.assertIn('detection_scores', output_blade)
        self.assertIn('detection_classes', output_blade)

        assert_array_almost_equal(
            output_jit['detection_boxes'],
            output_blade['detection_boxes'],
            decimal=3)
        assert_array_almost_equal(
            output_jit['detection_classes'],
            output_blade['detection_classes'],
            decimal=3)
        assert_array_almost_equal(
            output_jit['detection_scores'],
            output_blade['detection_scores'],
            decimal=3)

    def test_time(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        jit_path = PRETRAINED_MODEL_YOLOXS_EXPORT_JIT
        blade_path = PRETRAINED_MODEL_YOLOXS_EXPORT_BLADE

        predictor_jit = TorchYoloXPredictor(
            model_path=jit_path, score_thresh=0.5)

        predictor_blade = TorchYoloXPredictor(
            model_path=blade_path, score_thresh=0.5)

        input_data_list = [np.asarray(Image.open(img))]

        results = []

        results.append(
            benchmark(
                predictor_jit, input_data_list, model_name='easycv script'))
        results.append(
            benchmark(predictor_blade, input_data_list, model_name='blade'))

        logging.info('Model Summary:')
        summary = pd.DataFrame(results)
        logging.warning(summary.to_markdown())


if __name__ == '__main__':
    unittest.main()
