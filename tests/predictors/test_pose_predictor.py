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

from easycv.predictors.pose_predictor import TorchPoseTopDownPredictorWithDetector, vis_pose_result
from tests.ut_config import (PRETRAINED_MODEL_YOLOXS_EXPORT,
                             PRETRAINED_MODEL_POSE_HRNET_EXPORT,
                             POSE_DATA_SMALL_COCO_LOCAL)


class TorchPoseTopDownPredictorWithDetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_pose_topdown_with_detector(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        pose_model_path = PRETRAINED_MODEL_POSE_HRNET_EXPORT
        img = os.path.join(POSE_DATA_SMALL_COCO_LOCAL,
                           'images/000000067078.jpg')

        input_data_list = [np.asarray(Image.open(img))]
        model_path = ','.join((pose_model_path, detection_model_path))
        predictor = TorchPoseTopDownPredictorWithDetector(
            model_path=model_path,
            model_config={
                'pose': {
                    'bbox_thr': 0.3,
                    'format': 'xywh'
                },
                'detection': {
                    'model_type': 'TorchYoloXPredictor'
                }
            })

        all_pose_results = predictor.predict(input_data_list)
        one_result = all_pose_results[0]['pose_results']
        self.assertIn('bbox', one_result[1])
        self.assertIn('keypoints', one_result[1])
        self.assertEqual(len(one_result[1]['bbox']), 5)
        self.assertEqual(one_result[1]['keypoints'].shape, (17, 3))

        vis_result = vis_pose_result(
            predictor.pose_predictor.model,
            img,
            all_pose_results[0]['pose_results'],
            dataset_info=predictor.pose_predictor.dataset_info,
            show=False)

        vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)

        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            tmp_save_path = tmp_file.name
            cv2.imwrite(tmp_save_path, vis_result)
            assert os.path.exists(tmp_save_path)


if __name__ == '__main__':
    unittest.main()
