# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
from tests.ut_config import SMALL_COCO_WHOLE_BODY_HAND_ROOT

from easycv.datasets.pose.data_sources import HandCocoPoseTopDownSource

_DATA_CFG = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=21,
    num_joints=21,
    dataset_channel=[list(range(21))],
    inference_channel=list(range(21)),
)


class HandCocoPoseSourceCocoTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_top_down_source_coco(self):
        data_source = HandCocoPoseTopDownSource(
            data_cfg=_DATA_CFG,
            ann_file=
            f'{SMALL_COCO_WHOLE_BODY_HAND_ROOT}/annotations/small_whole_body_hand_coco.json',
            img_prefix=f'{SMALL_COCO_WHOLE_BODY_HAND_ROOT}/train2017/')
        index_list = random.choices(list(range(4)), k=3)
        for idx in index_list:
            data = data_source.get_sample(idx)
            self.assertIn('image_file', data)
            self.assertIn('image_id', data)
            self.assertIn('bbox_score', data)
            self.assertIn('bbox_id', data)
            self.assertIn('image_id', data)
            self.assertEqual(data['center'].shape, (2, ))
            self.assertEqual(data['scale'].shape, (2, ))
            self.assertEqual(len(data['bbox']), 4)
            self.assertEqual(data['joints_3d'].shape, (21, 3))
            self.assertEqual(data['joints_3d_visible'].shape, (21, 3))
            self.assertEqual(data['img'].shape[-1], 3)
            ann_info = data['ann_info']
            self.assertEqual(ann_info['image_size'].all(),
                             np.array([256, 256]).all())
            self.assertEqual(ann_info['heatmap_size'].all(),
                             np.array([64, 64]).all())
            self.assertEqual(ann_info['num_joints'], 21)
            self.assertEqual(len(ann_info['inference_channel']), 21)
            self.assertEqual(ann_info['num_output_channels'], 21)
            break

        self.assertEqual(len(data_source), 4)


if __name__ == '__main__':
    unittest.main()
