# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest
from os import path

import numpy as np
from tests.ut_config import POSE_DATA_MPII_DOWNLOAD_SMALL_LOCAL

from easycv.datasets.pose.data_sources.mpii import PoseTopDownSourceMpii

_DATA_CFG = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=16,
    num_joints=16,
    dataset_channel=[list(range(16))],
    inference_channel=list(range(16)),
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0)


class PoseTopDownSourceMpiiTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source, num):

        index_list = random.choices(list(range(10)), k=3)
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
            self.assertEqual(data['joints_3d'].shape, (16, 3))
            self.assertEqual(data['joints_3d_visible'].shape, (16, 3))
            self.assertEqual(data['img'].shape[-1], 3)
            ann_info = data['ann_info']
            self.assertEqual(ann_info['image_size'].all(),
                             np.array([288, 384]).all())
            self.assertEqual(ann_info['heatmap_size'].all(),
                             np.array([72, 96]).all())
            self.assertEqual(ann_info['num_joints'], 16)
            self.assertEqual(len(ann_info['inference_channel']), 16)
            self.assertEqual(ann_info['num_output_channels'], 16)
            self.assertEqual(len(ann_info['flip_pairs']), 11)
            self.assertEqual(len(ann_info['flip_pairs'][0]), 2)
            self.assertEqual(len(ann_info['flip_index']), 16)
            self.assertEqual(len(ann_info['upper_body_ids']), 9)
            self.assertEqual(len(ann_info['lower_body_ids']), 7)
            self.assertEqual(ann_info['joint_weights'].shape, (16, 1))
            self.assertEqual(len(ann_info['skeleton']), 16)
            self.assertEqual(len(ann_info['skeleton'][0]), 2)

        self.assertEqual(len(data_source), num)

    def test_top_down_source_mpii(self):
        CFG = {
            'annotaitions':
            'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/small_mpii/mpii_human_pose_v1_u12_2.zip',
            'images':
            'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/small_mpii/images.zip'
        }
        data_sourc_cfg = dict(
            path=POSE_DATA_MPII_DOWNLOAD_SMALL_LOCAL,
            download=True,
            test_mode=True,
            cfg=CFG,
            data_cfg=_DATA_CFG)

        data_source = PoseTopDownSourceMpii(**data_sourc_cfg)

        self._base_test(data_source, 29)


if __name__ == '__main__':
    unittest.main()
