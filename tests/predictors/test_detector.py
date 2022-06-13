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
                             PRETRAINED_MODEL_YOLOXS_END2END_JIT,
                             DET_DATA_SMALL_COCO_LOCAL)
from numpy.testing import assert_array_almost_equal


class DetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_yolox_detector(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT

        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        input_data_list = [np.asarray(Image.open(img))]
        predictor = TorchYoloXPredictor(
            model_path=detection_model_path, score_thresh=0.5)

        output = predictor.predict(input_data_list)[0]
        self.assertIn('detection_boxes', output)
        self.assertIn('detection_scores', output)
        self.assertIn('detection_classes', output)
        self.assertIn('detection_class_names', output)
        self.assertIn('ori_img_shape', output)
        self.assertEqual(len(output['detection_boxes']), 9)
        self.assertEqual(output['ori_img_shape'], [230, 352])

        self.assertListEqual(
            output['detection_classes'].tolist(),
            np.array([72, 69, 60, 56, 49, 49, 72, 46, 49],
                     dtype=np.int32).tolist())

        self.assertListEqual(output['detection_class_names'], [
            'refrigerator', 'oven', 'dining table', 'chair', 'orange',
            'orange', 'refrigerator', 'banana', 'orange'
        ])

        assert_array_almost_equal(
            output['detection_scores'],
            np.array([
                0.93252, 0.88439, 0.75048, 0.74093, 0.67255, 0.65550, 0.63942,
                0.60507, 0.56973
            ],
                     dtype=np.float32),
            decimal=2)

        assert_array_almost_equal(
            output['detection_boxes'],
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

    def test_yolox_detector_jit_end2end(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        jit_path = PRETRAINED_MODEL_YOLOXS_END2END_JIT
        predictor_jit = TorchYoloXPredictor(
            model_path=jit_path, score_thresh=0.5)

        output_jit = predictor_jit.predict(input_data_list)[0]

        self.assertIn('detection_boxes', output_jit)
        self.assertIn('detection_scores', output_jit)
        self.assertIn('detection_classes', output_jit)

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

    def test_yolox_detector_jit(self):
        img = os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                           'val2017/000000037777.jpg')

        input_data_list = [np.asarray(Image.open(img))]

        jit_path = PRETRAINED_MODEL_YOLOXS_EXPORT_JIT

        predictor_jit = TorchYoloXPredictor(
            model_path=jit_path, score_thresh=0.5)

        output_jit = predictor_jit.predict(input_data_list)[0]

        self.assertIn('detection_boxes', output_jit)
        self.assertIn('detection_scores', output_jit)
        self.assertIn('detection_classes', output_jit)

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


if __name__ == '__main__':
    unittest.main()
