# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import os.path as osp
import unittest

import mmcv
import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from tests.ut_config import POSE_DATA_SMALL_COCO_LOCAL
from xtcocotools.coco import COCO

from easycv.datasets.detection.pipelines import MMToTensor, NormalizeTensor
from easycv.datasets.pose.pipelines import (PoseCollect, TopDownAffine,
                                            TopDownGenerateTarget,
                                            TopDownGetRandomScaleRotation,
                                            TopDownHalfBodyTransform,
                                            TopDownRandomFlip,
                                            TopDownRandomTranslation)
from easycv.predictors.pose_predictor import _box2cs


def _check_flip(origin_imgs, result_imgs):
    """Check if the origin_imgs are flipped correctly."""
    h, w, c = origin_imgs.shape
    for i in range(h):
        for j in range(w):
            for k in range(c):
                if result_imgs[i, j, k] != origin_imgs[i, w - 1 - j, k]:
                    return False
    return True


def _check_normalize(origin_imgs, result_imgs, norm_cfg):
    """Check if the origin_imgs are normalized correctly into result_imgs in a
    given norm_cfg."""
    target_imgs = result_imgs.copy()
    for i in range(3):
        target_imgs[i] *= norm_cfg['std'][i]
        target_imgs[i] += norm_cfg['mean'][i]
    assert_array_almost_equal(origin_imgs, target_imgs, decimal=4)


class PoseTransformsTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_top_down_pipeline(self):
        # test loading
        data_prefix = POSE_DATA_SMALL_COCO_LOCAL
        ann_file = osp.join(data_prefix, 'train_200.json')
        coco = COCO(ann_file)

        results = dict(
            image_file=osp.join(data_prefix, 'images/000000472160.jpg'))
        results['img'] = mmcv.imread(results['image_file'], 'color', 'rgb')

        self.assertEqual(results['img'].shape, (513, 640, 3))
        image_size = (513, 640)

        ann_ids = coco.getAnnIds(472160)
        ann = coco.anns[ann_ids[0]]

        num_joints = 17
        joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
        joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
        for ipt in range(num_joints):
            joints_3d[ipt, 0] = ann['keypoints'][ipt * 3 + 0]
            joints_3d[ipt, 1] = ann['keypoints'][ipt * 3 + 1]
            joints_3d[ipt, 2] = 0
            t_vis = ann['keypoints'][ipt * 3 + 2]
            if t_vis > 1:
                t_vis = 1
            joints_3d_visible[ipt, 0] = t_vis
            joints_3d_visible[ipt, 1] = t_vis
            joints_3d_visible[ipt, 2] = 0

        center, scale = _box2cs(image_size, ann['bbox'])

        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['center'] = center
        results['scale'] = scale
        results['bbox_score'] = 1
        results['bbox_id'] = 0

        results['ann_info'] = {}
        results['ann_info']['flip_pairs'] = [[1, 2], [3, 4], [5, 6], [7, 8],
                                             [9, 10], [11, 12], [13, 14],
                                             [15, 16]]
        results['ann_info']['num_joints'] = num_joints
        results['ann_info']['upper_body_ids'] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10)
        results['ann_info']['lower_body_ids'] = (11, 12, 13, 14, 15, 16)
        results['ann_info']['use_different_joint_weights'] = False
        results['ann_info']['joint_weights'] = np.array(
            [
                1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2,
                1.2, 1.5, 1.5
            ],
            dtype=np.float32).reshape((num_joints, 1))
        results['ann_info']['image_size'] = np.array([192, 256])
        results['ann_info']['heatmap_size'] = np.array([48, 64])

        # test flip
        random_flip = TopDownRandomFlip(flip_prob=1.)
        results_flip = random_flip(copy.deepcopy(results))
        self.assertTrue(_check_flip(results['img'], results_flip['img']))

        # test random scale and rotate
        random_scale_rotate = TopDownGetRandomScaleRotation(90, 0.3, 1.0)
        results_scale_rotate = random_scale_rotate(copy.deepcopy(results))
        self.assertTrue(results_scale_rotate['rotation'] <= 180)
        self.assertTrue(results_scale_rotate['rotation'] >= -180)
        self.assertTrue(
            (results_scale_rotate['scale'] / results['scale'] <= 1.3).all())
        self.assertTrue(
            (results_scale_rotate['scale'] / results['scale'] >= 0.7).all())

        # test halfbody transform
        halfbody_transform = TopDownHalfBodyTransform(
            num_joints_half_body=8, prob_half_body=1.)
        results_halfbody = halfbody_transform(copy.deepcopy(results))
        self.assertTrue((results_halfbody['scale'] <= results['scale']).all())

        affine_transform = TopDownAffine()
        results['rotation'] = 90
        results_affine = affine_transform(copy.deepcopy(results))
        self.assertEqual(results_affine['img'].shape, (256, 192, 3))

        results = results_affine
        to_tensor = MMToTensor()
        results_tensor = to_tensor(copy.deepcopy(results))
        self.assertIsInstance(results_tensor['img'], torch.Tensor)
        self.assertEqual(results_tensor['img'].shape, torch.Size([3, 256,
                                                                  192]))

        norm_cfg = {}
        norm_cfg['mean'] = [0.485, 0.456, 0.406]
        norm_cfg['std'] = [0.229, 0.224, 0.225]

        normalize = NormalizeTensor(mean=norm_cfg['mean'], std=norm_cfg['std'])

        results_normalize = normalize(copy.deepcopy(results_tensor))
        _check_normalize(results_tensor['img'].data.numpy(),
                         results_normalize['img'].data.numpy(), norm_cfg)

        generate_target = TopDownGenerateTarget(
            sigma=2, target_type='GaussianHeatMap', unbiased_encoding=True)
        results_target = generate_target(copy.deepcopy(results_tensor))
        self.assertIn('target', results_target)
        self.assertEqual(results_target['target'].shape,
                         (num_joints, results['ann_info']['heatmap_size'][1],
                          results['ann_info']['heatmap_size'][0]))
        self.assertIn('target_weight', results_target)
        self.assertEqual(results_target['target_weight'].shape,
                         (num_joints, 1))

        generate_target = TopDownGenerateTarget(
            sigma=2, target_type='GaussianHeatmap', unbiased_encoding=True)
        results_target = generate_target(copy.deepcopy(results_tensor))
        self.assertIn('target', results_target)
        self.assertEqual(results_target['target'].shape,
                         (num_joints, results['ann_info']['heatmap_size'][1],
                          results['ann_info']['heatmap_size'][0]))
        self.assertIn('target_weight', results_target)
        self.assertEqual(results_target['target_weight'].shape,
                         (num_joints, 1))

        generate_target = TopDownGenerateTarget(
            sigma=2, unbiased_encoding=False)
        results_target = generate_target(copy.deepcopy(results_tensor))
        self.assertIn('target', results_target)
        self.assertEqual(results_target['target'].shape,
                         (num_joints, results['ann_info']['heatmap_size'][1],
                          results['ann_info']['heatmap_size'][0]))
        self.assertIn('target_weight', results_target)
        self.assertEqual(results_target['target_weight'].shape,
                         (num_joints, 1))

        generate_target = TopDownGenerateTarget(
            sigma=[2, 3], unbiased_encoding=False)
        results_target = generate_target(copy.deepcopy(results_tensor))
        self.assertIn('target', results_target)
        self.assertEqual(
            results_target['target'].shape,
            (2, num_joints, results['ann_info']['heatmap_size'][1],
             results['ann_info']['heatmap_size'][0]))
        self.assertIn('target_weight', results_target)
        self.assertEqual(results_target['target_weight'].shape,
                         (2, num_joints, 1))

        generate_target = TopDownGenerateTarget(
            kernel=(11, 11), encoding='Megvii', unbiased_encoding=False)
        results_target = generate_target(copy.deepcopy(results_tensor))
        self.assertIn('target', results_target)
        self.assertEqual(results_target['target'].shape,
                         (num_joints, results['ann_info']['heatmap_size'][1],
                          results['ann_info']['heatmap_size'][0]))
        self.assertIn('target_weight', results_target)
        self.assertEqual(results_target['target_weight'].shape,
                         (num_joints, 1))

        generate_target = TopDownGenerateTarget(
            kernel=[(11, 11), (7, 7)],
            encoding='Megvii',
            unbiased_encoding=False)
        results_target = generate_target(copy.deepcopy(results_tensor))
        self.assertIn('target', results_target)
        self.assertEqual(
            results_target['target'].shape,
            (2, num_joints, results['ann_info']['heatmap_size'][1],
             results['ann_info']['heatmap_size'][0]))
        self.assertIn('target_weight', results_target)
        self.assertEqual(results_target['target_weight'].shape,
                         (2, num_joints, 1))

        collect = PoseCollect(
            keys=['img', 'target', 'target_weight'],
            meta_keys=[
                'image_file', 'center', 'scale', 'rotation', 'bbox_score',
                'flip_pairs'
            ])
        results_final = collect(results_target)
        self.assertNotIn('img_size', results_final['img_metas'].data)
        self.assertIn('image_file', results_final['img_metas'].data)

    def test_random_translation(self):
        results = dict(
            center=np.zeros([2]),
            scale=1,
        )
        pipeline = TopDownRandomTranslation()
        results = pipeline(results)
        self.assertEqual(results['center'].shape, (2, ))


if __name__ == '__main__':
    unittest.main()
