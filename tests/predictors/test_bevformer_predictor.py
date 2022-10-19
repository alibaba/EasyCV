# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from tests.ut_config import (PRETRAINED_MODEL_BEVFORMER_BASE,
                             SMALL_NUSCENES_PATH)

import easycv
from easycv.predictors import BEVFormerPredictor


class BEVFormerPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _assert_results(self, results, assert_value=True):
        res = results['pts_bbox']
        self.assertEqual(res['scores_3d'].shape, torch.Size([300]))
        self.assertEqual(res['labels_3d'].shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].bev.shape, torch.Size([300, 5]))
        self.assertEqual(res['boxes_3d'].bottom_center.shape,
                         torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].bottom_height.shape,
                         torch.Size([300]))
        self.assertEqual(res['boxes_3d'].center.shape, torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].corners.shape, torch.Size([300, 8,
                                                                    3]))
        self.assertEqual(res['boxes_3d'].dims.shape, torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].gravity_center.shape,
                         torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].height.shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].nearest_bev.shape,
                         torch.Size([300, 4]))
        self.assertEqual(res['boxes_3d'].tensor.shape, torch.Size([300, 9]))
        self.assertEqual(res['boxes_3d'].top_height.shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].volume.shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].yaw.shape, torch.Size([300]))

        if not assert_value:
            return

        assert_array_almost_equal(
            res['scores_3d'][:5].numpy(),
            np.array([0.98207057, 0.9817677, 0.981756, 0.98154163, 0.98140806],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(res['labels_3d'][:10].numpy(),
                                  np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5]))
        assert_array_almost_equal(
            res['boxes_3d'].bev[:2].numpy(),
            np.array([[9.34029, -2.6638565, 2.0343924, 0.6568423, 1.8187382],
                      [6.6818047, -22.395258, 2.0680344, 0.6416968, 1.65906]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].bottom_center[:2].numpy(),
            np.array([[9.34029, -2.6638565, -1.8494891],
                      [6.6818047, -22.395258, -2.3321438]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].bottom_height[:5].numpy(),
            np.array(
                [-1.8494891, -2.3321438, -2.2945573, -1.5084305, -1.2044125],
                dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].center[:2].numpy(),
            np.array([[9.34029, -2.6638565, -1.8494891],
                      [6.6818047, -22.395258, -2.3321438]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].corners[:1][0][:3].numpy(),
            np.array([[9.908298, -3.5693488, -1.8494891],
                      [9.908298, -3.5693488, -0.7428185],
                      [9.271542, -3.730544, -0.7428185]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].dims[:2].numpy(),
            np.array([[2.0343924, 0.6568423, 1.1066706],
                      [2.0680344, 0.6416968, 1.0817788]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].gravity_center[:2].numpy(),
            np.array([[9.34029, -2.6638565, -1.2961538],
                      [6.6818047, -22.395258, -1.7912544]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].height[:5].numpy(),
            np.array([1.1066706, 1.0817788, 1.1004444, 1.0978955, 1.0731317],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].nearest_bev[:2].numpy(),
            np.array([[9.0118685, -3.6810527, 9.668712, -1.6466603],
                      [6.360956, -23.429276, 7.002653, -21.36124]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].tensor[:1].numpy(),
            np.array([[
                9.3402901e+00, -2.6638565e+00, -1.8494891e+00, 2.0343924e+00,
                6.5684229e-01, 1.1066706e+00, 1.8187382e+00, -8.6360151e-06,
                2.0341220e-05
            ]],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].top_height[:5].numpy(),
            np.array(
                [-0.7428185, -1.250365, -1.1941129, -0.41053498, -0.13128078],
                dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].volume[:5].numpy(),
            np.array([1.4788163, 1.4355756, 1.490336, 1.49576, 1.4706278],
                     dtype=np.float32),
            decimal=4)
        assert_array_almost_equal(
            res['boxes_3d'].yaw[:5].numpy(),
            np.array([1.8187382, 1.65906, 1.694045, 1.6197505, 1.6418235],
                     dtype=np.float32),
            decimal=4)

    def test_single(self):
        model_path = PRETRAINED_MODEL_BEVFORMER_BASE
        single_ann_file = os.path.join(SMALL_NUSCENES_PATH,
                                       'inference/single_sample.pkl')
        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(
            config_dir,
            'detection3d/bevformer/bevformer_base_r101_dcn_nuscenes.py')

        predictor = BEVFormerPredictor(
            model_path=model_path,
            config_file=config_file,
        )
        results = predictor(single_ann_file)
        self.assertEqual(len(results), 1)
        for result in results:
            self._assert_results(result)

    def test_batch(self):
        model_path = PRETRAINED_MODEL_BEVFORMER_BASE
        single_ann_file = os.path.join(SMALL_NUSCENES_PATH,
                                       'inference/single_sample.pkl')
        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(
            config_dir,
            'detection3d/bevformer/bevformer_base_r101_dcn_nuscenes.py')

        predictor = BEVFormerPredictor(
            model_path=model_path, config_file=config_file, batch_size=1)
        results = predictor([single_ann_file, single_ann_file])
        self.assertEqual(len(results), 2)
        # Input the same sample continuously, the output value is different,
        # because the model will record the features of the previous sample to infer the next sample
        self._assert_results(results[0])
        self._assert_results(results[1], assert_value=False)


if __name__ == '__main__':
    unittest.main()
