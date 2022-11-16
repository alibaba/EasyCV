# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
from tests.ut_config import POSE_DATA_OC_HUMAN_SMALL_LOCAL

from easycv.datasets.pose.data_sources.oc_human import PoseTopDownSourceChHuman

_DATA_CFG = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=17,
    num_joints=17,
    dataset_channel=[list(range(17))],
    inference_channel=list(range(17)),
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=True,
    det_bbox_thr=0.0)


class PoseTopDownSourceCrowdPoseTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source, num):

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
            self.assertEqual(data['joints_3d'].shape, (17, 3))
            self.assertEqual(data['joints_3d_visible'].shape, (17, 3))
            self.assertEqual(data['img'].shape[-1], 3)
            ann_info = data['ann_info']
            self.assertEqual(ann_info['image_size'].all(),
                             np.array([288, 384]).all())
            self.assertEqual(ann_info['heatmap_size'].all(),
                             np.array([72, 96]).all())
            self.assertEqual(ann_info['num_joints'], 17)
            self.assertEqual(len(ann_info['inference_channel']), 17)
            self.assertEqual(ann_info['num_output_channels'], 17)
            self.assertEqual(len(ann_info['flip_pairs']), 8)
            self.assertEqual(len(ann_info['flip_pairs'][0]), 2)
            self.assertEqual(len(ann_info['flip_index']), 17)
            self.assertEqual(len(ann_info['upper_body_ids']), 11)
            self.assertEqual(len(ann_info['lower_body_ids']), 6)
            self.assertEqual(ann_info['joint_weights'].shape, (17, 1))
            self.assertEqual(len(ann_info['skeleton']), 19)
            self.assertEqual(len(ann_info['skeleton'][0]), 2)

        self.assertEqual(len(data_source), num)

    def test_top_down_source_oc_human(self):
        data_sourc_cfg_1 = dict(
            ann_file=POSE_DATA_OC_HUMAN_SMALL_LOCAL + 'ochuman.json',
            img_prefix=POSE_DATA_OC_HUMAN_SMALL_LOCAL + 'images',
            test_mode=True,
            subset='train',
            data_cfg=_DATA_CFG)

        data_sourc_cfg_2 = dict(
            ann_file=POSE_DATA_OC_HUMAN_SMALL_LOCAL +
            'ochuman_coco_format_20.json',
            img_prefix=POSE_DATA_OC_HUMAN_SMALL_LOCAL + 'images',
            test_mode=True,
            subset=None,
            data_cfg=_DATA_CFG)
        data_source_1 = PoseTopDownSourceChHuman(**data_sourc_cfg_1)
        data_source_2 = PoseTopDownSourceChHuman(**data_sourc_cfg_2)
        self._base_test(data_source_1, 30)
        self._base_test(data_source_2, 32)


if __name__ == '__main__':
    unittest.main()
