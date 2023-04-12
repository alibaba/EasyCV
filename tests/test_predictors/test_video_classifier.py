# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import os
import unittest
import numpy as np

from easycv.predictors.video_classifier import VideoClassificationPredictor, STGCNPredictor
from tests.ut_config import (PRETRAINED_MODEL_X3D_XS,
                             VIDEO_DATA_SMALL_RAW_LOCAL, BASE_LOCAL_PATH)


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


class STGCNPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_single(self):
        checkpoint = os.path.join(
            BASE_LOCAL_PATH,
            'pretrained_models/video/stgcn/stgcn_80e_ntu60_xsub.pth')
        config_file = 'configs/video_recognition/stgcn/stgcn_80e_ntu60_xsub_keypoint.py'
        predict_op = STGCNPredictor(
            model_path=checkpoint, config_file=config_file)

        h, w = 480, 853
        total_frames = 20
        num_person = 2
        inp = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=total_frames,
            keypoint=np.random.random((num_person, total_frames, 17, 2)),
            keypoint_score=np.random.random((num_person, total_frames, 17)),
        )

        results = predict_op([inp])[0]
        self.assertIn('class', results)
        self.assertIn('class_name', results)
        self.assertEqual(len(results['class_probs']), 60)

    def test_jit(self):
        checkpoint = os.path.join(
            BASE_LOCAL_PATH,
            'pretrained_models/video/stgcn/stgcn_80e_ntu60_xsub.pth.jit')

        config_file = 'configs/video_recognition/stgcn/stgcn_80e_ntu60_xsub_keypoint.py'
        predict_op = STGCNPredictor(
            model_path=checkpoint, config_file=config_file)

        h, w = 480, 853
        total_frames = 20
        num_person = 2
        inp = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=total_frames,
            keypoint=np.random.random((num_person, total_frames, 17, 2)),
            keypoint_score=np.random.random((num_person, total_frames, 17)),
        )

        results = predict_op([inp])[0]
        self.assertIn('class', results)
        self.assertIn('class_name', results)
        self.assertEqual(len(results['class_probs']), 60)


if __name__ == '__main__':
    unittest.main()
