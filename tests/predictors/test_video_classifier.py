# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import json
import os
import unittest

import cv2
import torch
from easycv.predictors.video_classifier import VideoClassificationPredictor
from easycv.utils.test_util import clean_up, get_tmp_dir
from tests.ut_config import (PRETRAINED_MODEL_X3D_XS,
                             VIDEO_DATA_SMALL_RAW_LOCAL)


class VideoClassificationPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_single(self):
        checkpoint = PRETRAINED_MODEL_X3D_XS
        config_file = 'configs/video_recognition/x3d/x3d_xs.py'
        predict_op = VideoClassificationPredictor(
            model_path=checkpoint, config_file=config_file)
        img_path = os.path.join(
            VIDEO_DATA_SMALL_RAW_LOCAL,
            'kinetics400/val_256/y5xuvHpDPZQ_000005_000015.mp4')

        input = {'filename': img_path}
        results = predict_op([input])[0]
        self.assertListEqual(results['class'], [55])
        self.assertListEqual(results['class_name'], ['55'])
        self.assertEqual(len(results['class_probs']), 400)

    def test_batch(self):
        checkpoint = PRETRAINED_MODEL_X3D_XS
        config_file = 'configs/video_recognition/x3d/x3d_xs.py'
        predict_op = VideoClassificationPredictor(
            model_path=checkpoint, config_file=config_file)
        img_path = os.path.join(
            VIDEO_DATA_SMALL_RAW_LOCAL,
            'kinetics400/val_256/y5xuvHpDPZQ_000005_000015.mp4')

        input = {'filename': img_path}

        num_imgs = 4
        results = predict_op([input] * num_imgs)
        self.assertEqual(len(results), num_imgs)
        for res in results:
            # self.assertListEqual(res['class'], [55])
            # self.assertListEqual(res['class_name'], ['55'])
            self.assertEqual(len(res['class_probs']), 400)


if __name__ == '__main__':
    unittest.main()
