# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

import cv2
from ut_config import PRETRAINED_MODEL_FACE_2D_KEYPOINTS

from easycv.predictors.face_keypoints_predictor import FaceKeypointsPredictor


class FaceKeypointsPredictorWithoutDetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.image_path = './data/test/face_2d_keypoints/data/002258.png'
        self.save_image_path = './data/test/face_2d_keypoints/data/result_002258.png'
        self.model_path = PRETRAINED_MODEL_FACE_2D_KEYPOINTS
        self.model_config_path = './configs/face/face_96x96_wingloss.py'

    def test_single(self):
        predict_pipeline = FaceKeypointsPredictor(
            model_path=self.model_path, config_file=self.model_config_path)
        output = predict_pipeline(self.image_path)[0]
        output_keypoints = output['point']
        output_pose = output['pose']
        img = cv2.imread(self.image_path)
        image_show = predict_pipeline.show_result(
            img, output_keypoints, scale=2, save_path=self.save_image_path)
        self.assertEqual(output_keypoints.shape[0], 106)
        self.assertEqual(output_keypoints.shape[1], 2)
        self.assertEqual(output_pose.shape[0], 3)

    def test_batch(self):
        predict_pipeline = FaceKeypointsPredictor(
            model_path=self.model_path,
            config_file=self.model_config_path,
            batch_size=2)

        total_samples = 3
        output = predict_pipeline([self.image_path] * total_samples)

        self.assertEqual(len(output), total_samples)
        for out in output:
            self.assertEqual(out['point'].shape, (106, 2))
            self.assertEqual(out['pose'].shape, (3, ))


if __name__ == '__main__':
    unittest.main()
