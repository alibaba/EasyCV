# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
from tests.ut_config import (COCO_DATASET_DOWNLOAD_SMALL,
                             DET_DATA_COCO2017_DOWNLOAD,
                             POSE_DATA_SMALL_COCO_LOCAL)

from easycv.datasets.pose.data_sources.coco import (PoseTopDownSourceCoco,
                                                    PoseTopDownSourceCoco2017)
from easycv.datasets.utils.download_data.download_coco import download_coco

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
    use_gt_bbox=False,
    det_bbox_thr=0.0)


class PoseTopDownSourceCocoTest(unittest.TestCase):

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

            break

        self.assertEqual(len(data_source), 420)

    def test_top_down_source_coco(self):
        data_source = PoseTopDownSourceCoco(
            data_cfg=_DATA_CFG,
            ann_file=f'{POSE_DATA_SMALL_COCO_LOCAL}/train_200.json',
            img_prefix=f'{POSE_DATA_SMALL_COCO_LOCAL}/images/')

        self._base_test(data_source)

    def test_top_down_source_coco_2017(self):
        data_source = PoseTopDownSourceCoco2017(
            data_cfg=_DATA_CFG,
            path=DET_DATA_COCO2017_DOWNLOAD,
            download=True,
            split='train')

        self._base_test(data_source)

    def test_download_coco(self):
        cfg = dict(
            coco2017=[
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/annotations.zip',
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/train2017_small.zip',
                'https://easycv.oss-cn-hangzhou.aliyuncs.com/data/val2017_small.zip',
            ],
            detection=dict(
                train='instances_train2017.json',
                val='instances_val2017.json',
            ),
            train_dataset='train2017',
            val_dataset='val2017',
            pose=dict(
                train='person_keypoints_train2017.json',
                val='person_keypoints_val2017.json'))

        coco_path = download_coco(
            'coco2017',
            'train',
            target_dir=COCO_DATASET_DOWNLOAD_SMALL,
            task='pose',
            cfg=cfg)
        data_source = PoseTopDownSourceCoco(
            data_cfg=_DATA_CFG,
            ann_file=coco_path['ann_file'],
            img_prefix=coco_path['img_prefix'])

        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
