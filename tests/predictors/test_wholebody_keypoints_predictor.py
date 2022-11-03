# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from ut_config import (PRETRAINED_MODEL_WHOLEBODY,
                       PRETRAINED_MODEL_WHOLEBODY_DETECTION)

from easycv.predictors.wholebody_keypoints_predictor import \
    WholeBodyKeypointsPredictor

DEFAULT_WHOLEBODY_DETECTION_MODEL_PATH = PRETRAINED_MODEL_WHOLEBODY_DETECTION
DEFAULT_WHOLEBODY_DETECTION_CONFIG_FILE = 'configs/detection/yolox/yolox_x_8xb8_300e_coco.py'


class WholeBodyKeypointsPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.image_path = 'data/test/pose/wholebody/data/img_test_wholebody.jpg'
        self.save_image_path = 'img_test_wholebody_ret.jpg'
        self.model_path = PRETRAINED_MODEL_WHOLEBODY
        self.model_config_path = 'configs/pose/wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'

    def test_single(self):
        predict_pipeline = WholeBodyKeypointsPredictor(
            model_path=self.model_path,
            config_file=self.model_config_path,
            detection_predictor_config=dict(
                type='DetectionPredictor',
                model_path=DEFAULT_WHOLEBODY_DETECTION_MODEL_PATH,
                config_file=DEFAULT_WHOLEBODY_DETECTION_CONFIG_FILE,
                score_threshold=0.5),
            bbox_thr=0.8)

        output = predict_pipeline(self.image_path)[0]
        keypoints = output['keypoints']
        boxes = output['boxes']

        image_show = predict_pipeline.show_result(
            self.image_path,
            keypoints,
            boxes,
            scale=1,
            save_path=self.save_image_path)

        for keypoint in keypoints:
            self.assertEqual(keypoint.shape[0], 133)
        for box in boxes:
            self.assertEqual(box.shape[0], 4)

    def test_batch(self):
        predict_pipeline = WholeBodyKeypointsPredictor(
            model_path=self.model_path,
            config_file=self.model_config_path,
            detection_predictor_config=dict(
                type='DetectionPredictor',
                model_path=DEFAULT_WHOLEBODY_DETECTION_MODEL_PATH,
                config_file=DEFAULT_WHOLEBODY_DETECTION_CONFIG_FILE,
                score_threshold=0.5),
            bbox_thr=0.8,
            batch_size=2)

        total_samples = 3
        output = predict_pipeline([self.image_path] * total_samples)

        self.assertEqual(len(output), 2)
        for out in output:
            keypoints = out['keypoints']
            boxes = out['boxes']
            for keypoint in keypoints:
                self.assertEqual(keypoint.shape[0], 133)
            for box in boxes:
                self.assertEqual(box.shape[0], 4)


if __name__ == '__main__':
    unittest.main()
