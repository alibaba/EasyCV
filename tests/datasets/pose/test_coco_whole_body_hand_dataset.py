# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from tests.ut_config import SMALL_COCO_WHOLE_BODY_HAND_ROOT

from easycv.datasets.pose import HandCocoWholeBodyDataset

_DATA_CFG = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=21,
    num_joints=21,
    dataset_channel=[list(range(21))],
    inference_channel=list(range(21)))

_DATASET_ARGS = [{
    'data_source':
    dict(
        type='HandCocoPoseTopDownSource',
        data_cfg=_DATA_CFG,
        ann_file=
        f'{SMALL_COCO_WHOLE_BODY_HAND_ROOT}/annotations/small_whole_body_hand_coco.json',
        img_prefix=f'{SMALL_COCO_WHOLE_BODY_HAND_ROOT}/train2017/'),
    'pipeline': [
        dict(type='TopDownRandomFlip', flip_prob=0.5),
        dict(type='TopDownAffine'),
        dict(type='MMToTensor'),
        dict(type='TopDownGenerateTarget', sigma=3),
        dict(
            type='PoseCollect',
            keys=['img', 'target', 'target_weight'],
            meta_keys=[
                'image_file', 'joints_3d', 'flip_pairs', 'joints_3d_visible',
                'center', 'scale', 'rotation', 'bbox_score'
            ])
    ]
}, {}]


class PoseTopDownDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    @staticmethod
    def build_dataset(index):
        dataset = HandCocoWholeBodyDataset(
            data_source=_DATASET_ARGS[index].get('data_source', None),
            pipeline=_DATASET_ARGS[index].get('pipeline', None))

        return dataset

    def test_0(self, index=0):
        dataset = self.build_dataset(index)
        ann_info = dataset.data_source.ann_info

        self.assertEqual(len(dataset), 4)
        for i, batch in enumerate(dataset):
            self.assertEqual(
                batch['img'].shape,
                torch.Size([3] + list(ann_info['image_size'][::-1])))
            self.assertEqual(batch['target'].shape,
                             (ann_info['num_joints'], ) +
                             tuple(ann_info['heatmap_size'][::-1]))
            self.assertEqual(batch['img_metas'].data['joints_3d'].shape,
                             (ann_info['num_joints'], 3))
            self.assertIn('center', batch['img_metas'].data)
            self.assertIn('scale', batch['img_metas'].data)

            break


if __name__ == '__main__':
    unittest.main()
