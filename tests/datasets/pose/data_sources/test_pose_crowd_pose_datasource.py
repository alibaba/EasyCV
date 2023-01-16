# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
from tests.ut_config import POSE_DATA_CROWDPOSE_SMALL_LOCAL

from easycv.datasets.pose.data_sources.crowd_pose import \
    PoseTopDownSourceCrowdPose

_DATA_CFG = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=14,
    num_joints=14,
    dataset_channel=[list(range(14))],
    inference_channel=list(range(14)),
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0)


class PoseTopDownSourceCrowdPoseTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source):

        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source[idx]
            self.assertIn('image_file', data)
            self.assertIn('image_id', data)
            self.assertIn('bbox_score', data)
            self.assertIn('bbox_id', data)
            self.assertIn('image_id', data)
            self.assertEqual(data['center'].shape, (2, ))
            self.assertEqual(data['scale'].shape, (2, ))
            self.assertEqual(len(data['bbox']), 4)
            self.assertEqual(data['joints_3d'].shape, (14, 3))
            self.assertEqual(data['joints_3d_visible'].shape, (14, 3))
            self.assertEqual(data['img'].shape[-1], 3)
            ann_info = data['ann_info']
            self.assertEqual(ann_info['image_size'].all(),
                             np.array([288, 384]).all())
            self.assertEqual(ann_info['heatmap_size'].all(),
                             np.array([72, 96]).all())
            self.assertEqual(ann_info['num_joints'], 14)
            self.assertEqual(len(ann_info['inference_channel']), 14)
            self.assertEqual(ann_info['num_output_channels'], 14)
            self.assertEqual(len(ann_info['flip_pairs']), 10)
            self.assertEqual(len(ann_info['flip_pairs'][0]), 2)
            self.assertEqual(len(ann_info['flip_index']), 14)
            self.assertEqual(len(ann_info['upper_body_ids']), 8)
            self.assertEqual(len(ann_info['lower_body_ids']), 6)
            self.assertEqual(ann_info['joint_weights'].shape, (14, 1))
            self.assertEqual(len(ann_info['skeleton']), 13)
            self.assertEqual(len(ann_info['skeleton'][0]), 2)

        self.assertEqual(len(data_source), 62)

    def test_top_down_source_coco_2017(self):
        data_sourc_cfg = dict(
            ann_file=POSE_DATA_CROWDPOSE_SMALL_LOCAL + 'train20.json',
            img_prefix=POSE_DATA_CROWDPOSE_SMALL_LOCAL + 'images',
            test_mode=True,
            data_cfg=_DATA_CFG)
        data_source = PoseTopDownSourceCrowdPose(**data_sourc_cfg)
        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
