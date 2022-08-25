# Copyright (c) Alibaba, Inc. and its affiliates.

import copy
import os
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image

from easycv.predictors.face_keypoints_predictor import FaceKeypointsPredictor


class FaceKeypointsPredictorWithoutDetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.image_path = './data/test/face_2d_keypoints/data/002253.png'
        self.save_image_path = './data/test/face_2d_keypoints/data/result_002253.png'
        self.model_path = './data/test/face_2d_keypoints/models/epoch_580.pth'
        self.model_config_path = './configs/face/face_96x96_wingloss.py'

    def test_single(self):
        predict_pipeline = FaceKeypointsPredictor(
            model_path=self.model_path, model_config=self.model_config_path)

        output = predict_pipeline(self.image_path)[0]
        output_keypoints = output['point']
        output_pose = output['pose']
        image_show = predict_pipeline.show_result(
            self.image_path,
            output_keypoints,
            scale=2,
            save_path=self.save_image_path)


if __name__ == '__main__':
    unittest.main()
