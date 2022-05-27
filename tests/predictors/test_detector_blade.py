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

from easycv.utils.test_util import benchmark
import logging
import pandas as pd
import torch
from numpy.testing import assert_array_almost_equal


@unittest.skipIf(torch.__version__!='1.8.1+cu102',
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

        self.assertListEqual(
            output_jit['detection_classes'].tolist(),
            np.array([72, 69, 60, 56, 49, 49, 72, 46, 49],
                     dtype=np.int32).tolist())

        self.assertListEqual(output_jit['detection_class_names'], [
            'refrigerator', 'oven', 'dining table', 'chair', 'orange',
            'orange', 'refrigerator', 'banana', 'orange'
        ])

        assert_array_almost_equal(
            output_jit['detection_scores'],
            np.array([
                0.93252, 0.88439, 0.75048, 0.74093, 0.67255, 0.65550, 0.63942,
                0.60507, 0.56973
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output_jit['detection_boxes'],
            np.array([[298.28256, 76.26037, 352., 228.91579],
                      [137.9849, 124.92237, 196.49876, 193.12375],
                      [76.42237, 170.30052, 292.4093, 227.32962],
                      [117.317, 188.9916, 165.43694, 212.3457],
                      [231.36719, 199.89865, 248.27888, 217.50288],
                      [217.1154, 200.18729, 232.20607, 214.38866],
                      [121.948105, 90.01667, 193.17673, 194.04584],
                      [240.4494, 188.07112, 258.7406, 206.78226],
                      [204.21452, 187.11292, 220.3842, 207.25877]]),
            decimal=1)

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

        self.assertListEqual(
            output_jit['detection_classes'].tolist(),
            np.array([72, 69, 60, 56, 49, 49, 72, 46, 49],
                     dtype=np.int32).tolist())

        self.assertListEqual(output_jit['detection_class_names'], [
            'refrigerator', 'oven', 'dining table', 'chair', 'orange',
            'orange', 'refrigerator', 'banana', 'orange'
        ])

        assert_array_almost_equal(
            output_jit['detection_scores'],
            np.array([
                0.93252, 0.88439, 0.75048, 0.74093, 0.67255, 0.65550, 0.63942,
                0.60507, 0.56973
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output_jit['detection_boxes'],
            np.array([[298.28256, 76.26037, 352., 228.91579],
                      [137.9849, 124.92237, 196.49876, 193.12375],
                      [76.42237, 170.30052, 292.4093, 227.32962],
                      [117.317, 188.9916, 165.43694, 212.3457],
                      [231.36719, 199.89865, 248.27888, 217.50288],
                      [217.1154, 200.18729, 232.20607, 214.38866],
                      [121.948105, 90.01667, 193.17673, 194.04584],
                      [240.4494, 188.07112, 258.7406, 206.78226],
                      [204.21452, 187.11292, 220.3842, 207.25877]]),
            decimal=1)

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
