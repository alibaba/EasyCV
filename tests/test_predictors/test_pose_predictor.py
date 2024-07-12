# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import tempfile
import unittest

import cv2
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image
from tests.ut_config import (BASE_LOCAL_PATH, POSE_DATA_SMALL_COCO_LOCAL,
                             PRETRAINED_MODEL_POSE_HRNET_EXPORT,
                             PRETRAINED_MODEL_YOLOXS_EXPORT, TEST_IMAGES_DIR)

from easycv.predictors.pose_predictor import (
    PoseTopDownPredictor, TorchPoseTopDownPredictorWithDetector)


class PoseTopDownPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, predictor):
        img1 = os.path.join(POSE_DATA_SMALL_COCO_LOCAL,
                            'images/000000067078.jpg')
        img2 = os.path.join(TEST_IMAGES_DIR, 'crowdpose_100024.jpg')
        input_data_list = [img1, img2]

        results = predictor(input_data_list)
        self.assertEqual(len(results), 2)

        result0 = results[0]
        self.assertEqual(result0['bbox'].shape, (4, 5))
        self.assertEqual(result0['keypoints'].shape, (4, 17, 3))

        assert_array_almost_equal(
            result0['keypoints'][0][0], [509.8026, 111.99933, 0.9709578],
            decimal=1)
        assert_array_almost_equal(
            result0['keypoints'][0][9], [561.235, 312.41324, 0.9236345],
            decimal=1)
        assert_array_almost_equal(
            result0['keypoints'][1][1], [55.37381, 196.2315, 0.9558682],
            decimal=1)
        assert_array_almost_equal(
            result0['keypoints'][1][12], [47.469627, 297.25607, 0.5480971],
            decimal=1)
        assert_array_almost_equal(
            result0['keypoints'][3][5], [293.57898, 166.9432, 0.8903505],
            decimal=1)
        assert_array_almost_equal(
            result0['keypoints'][3][10], [264.51807, 178.30908, 0.920545],
            decimal=1)

        assert_array_almost_equal(
            result0['bbox'],
            np.array([[438.9, 59., 604.8, 511.2, 0.9],
                      [10.5, 179.6, 101.8, 297.7, 0.9],
                      [229.6, 114.4, 307.8, 231.4, 0.6],
                      [229.4, 114.7, 308.5, 296.7, 0.6]],
                     dtype=np.float32),
            decimal=1)
        vis_result = predictor.show_result(img1, result0)

        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            tmp_save_path = tmp_file.name
            # tmp_save_path = './show1.jpg'
            cv2.imwrite(tmp_save_path, vis_result)
            assert os.path.exists(tmp_save_path)

        result1 = results[1]
        self.assertEqual(result1['bbox'].shape, (15, 5))
        self.assertEqual(result1['keypoints'].shape, (15, 17, 3))

        assert_array_almost_equal(
            result1['keypoints'][0][0], [510.86044, 234.81412, 0.42776352],
            decimal=1)
        assert_array_almost_equal(
            result1['keypoints'][0][8], [537.6073, 288.58582, 0.92016876],
            decimal=1)
        assert_array_almost_equal(
            result1['keypoints'][2][1], [191.784, 114.456, 0.963], decimal=1)
        assert_array_almost_equal(
            result1['keypoints'][2][15], [200.38428, 247.03822, 0.9013438],
            decimal=1)
        assert_array_almost_equal(
            result1['keypoints'][8][8], [153.85138, 1.5582924, 0.91658807],
            decimal=1)
        assert_array_almost_equal(
            result1['keypoints'][13][6], [475.59854, 50.610546, 0.29660743],
            decimal=1)

        assert_array_almost_equal(
            result1['bbox'][:4],
            np.array([[470.6, 214.7, 549.9, 412.1, 0.9],
                      [71.6, 91., 136.7, 251.4, 0.9],
                      [159.7, 100.4, 221.9, 269.4, 0.9],
                      [219.4, 117.3, 281.9, 278.8, 0.9]],
                     dtype=np.float32),
            decimal=1)
        vis_result = predictor.show_result(img2, result1)

        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            tmp_save_path = tmp_file.name
            # tmp_save_path = './show2.jpg'
            cv2.imwrite(tmp_save_path, vis_result)
            assert os.path.exists(tmp_save_path)

    def test_pose_topdown(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        pose_model_path = PRETRAINED_MODEL_POSE_HRNET_EXPORT

        predictor = PoseTopDownPredictor(
            model_path=pose_model_path,
            detection_predictor_config=dict(
                type='YoloXPredictor',
                model_path=detection_model_path,
                score_thresh=0.5),
            cat_id=0,
            batch_size=1)

        self._base_test(predictor)

    def test_pose_topdown_jit(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        pose_model_path = os.path.join(
            BASE_LOCAL_PATH,
            'pretrained_models/pose/hrnet/pose_hrnet_epoch_210_export.pth.jit')

        config_file = 'configs/pose/hrnet_w48_coco_256x192_udp.py'

        predictor = PoseTopDownPredictor(
            model_path=pose_model_path,
            config_file=config_file,
            detection_predictor_config=dict(
                type='YoloXPredictor',
                model_path=detection_model_path,
                score_thresh=0.5),
            cat_id=0,
            batch_size=1)

        img = os.path.join(TEST_IMAGES_DIR, 'im00025.png')
        input_data_list = [img, img]
        results = predictor(input_data_list)
        self.assertEqual(len(results), 2)


class TorchPoseTopDownPredictorWithDetectorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_pose_topdown_with_detector(self):
        detection_model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        pose_model_path = PRETRAINED_MODEL_POSE_HRNET_EXPORT
        # img = os.path.join(POSE_DATA_SMALL_COCO_LOCAL,
        #                    'images/000000067078.jpg')
        img = os.path.join(TEST_IMAGES_DIR, 'crowdpose_100024.jpg')
        input_data_list = [np.asarray(Image.open(img))]
        model_path = ','.join((pose_model_path, detection_model_path))
        predictor = TorchPoseTopDownPredictorWithDetector(
            model_path=model_path,
            model_config={
                'pose': {
                    'bbox_thr': 0.5,
                    'format': 'xywh'
                },
                'detection': {
                    'model_type': 'TorchYoloXPredictor',
                    'reserved_classes': ['person'],
                }
            })

        all_pose_results = predictor(input_data_list)
        one_result = all_pose_results[0]['pose_results']
        self.assertIn('bbox', one_result[1])
        self.assertIn('keypoints', one_result[1])
        self.assertEqual(len(one_result[1]['bbox']), 5)
        self.assertEqual(one_result[1]['keypoints'].shape, (17, 3))

        vis_result = predictor.show_result(
            img, all_pose_results[0]['pose_results'], show=False)

        # vis_result = cv2.resize(vis_result, dsize=None, fx=0.5, fy=0.5)

        with tempfile.NamedTemporaryFile(suffix='.jpg') as tmp_file:
            tmp_save_path = tmp_file.name
            # tmp_save_path = './show3.jpg'
            cv2.imwrite(tmp_save_path, vis_result)
            assert os.path.exists(tmp_save_path)


if __name__ == '__main__':
    unittest.main()
