# Copyright (c) Alibaba, Inc. and its affiliates.

import unittest

from easycv.predictors.hand_keypoints_predictor import HandKeypointsPredictor
from easycv.utils.config_tools import mmcv_config_fromfile

MM_DEFAULT_HAND_DETECTION_SSDLITE_MODEL_PATH = 'https://download.openmmlab.com/mmpose/mmdet_pretrained/' \
                                               'ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'
MM_DEFAULT_HAND_DETECTION_SSDLITE_CONFIG_FILE = 'data/test/pose/hand/configs/hand_keypoints_predictor.py'


class HandKeypointsPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.image_path = 'data/test/pose/hand/data/hand.jpg'
        self.save_image_path = 'data/test/pose/hand/data/hand_result.jpg'
        self.model_path = 'data/test/pose/hand/hrnet_w18_256x256.pth'
        self.model_config_path = 'configs/pose/hand/hrnet_w18_coco_wholebody_hand_256x256_dark.py'

    def test_single(self):
        config = mmcv_config_fromfile(self.model_config_path)
        predict_pipeline = HandKeypointsPredictor(
            model_path=self.model_path,
            config_file=config,
            detection_predictor_config=dict(
                type='DetectionPredictor',
                model_path=MM_DEFAULT_HAND_DETECTION_SSDLITE_MODEL_PATH,
                config_file=MM_DEFAULT_HAND_DETECTION_SSDLITE_CONFIG_FILE,
                score_threshold=0.5))

        output = predict_pipeline(self.image_path)[0]
        keypoints = output['keypoints']
        boxes = output['boxes']
        image_show = predict_pipeline.show_result(
            self.image_path, keypoints, boxes, save_path=self.save_image_path)
        self.assertEqual(keypoints.shape[0], 1)
        self.assertEqual(keypoints.shape[1], 21)
        self.assertEqual(keypoints.shape[2], 3)


if __name__ == '__main__':
    unittest.main()
