# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from tests.ut_config import SMALL_COCO_WHOLEBODY_ROOT

from easycv.datasets.builder import build_datasource
from easycv.datasets.pose import WholeBodyCocoTopDownDataset

channel_cfg = dict(
    num_output_channels=133,
    dataset_joints=133,
    dataset_channel=[
        list(range(133)),
    ],
    inference_channel=list(range(133)))

data_cfg = dict(
    image_size=[288, 384],
    heatmap_size=[72, 96],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
)

data_source_cfg = dict(
    type='WholeBodyCocoTopDownSource',
    data_cfg=data_cfg,
    ann_file=
    f'{SMALL_COCO_WHOLEBODY_ROOT}/annotations/test_coco_wholebody.json',
    img_prefix=f'{SMALL_COCO_WHOLEBODY_ROOT}/train2017/')

pipeline = [
    dict(type='TopDownGetBboxCenterScale', padding=1.25),
    dict(type='TopDownRandomShiftBboxCenter', shift_factor=0.16, prob=0.3),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.5),
    dict(type='TopDownAffine'),
    dict(type='MMToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(type='TopDownGenerateTarget', sigma=3, unbiased_encoding=True),
    dict(
        type='PoseCollect',
        keys=['img', 'target', 'target_weight'],
        meta_keys=[
            'image_file', 'image_id', 'joints_3d', 'joints_3d_visible',
            'center', 'scale', 'rotation', 'flip_pairs'
        ])
]


class WholeBodyTopDownCocoDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        dataset = WholeBodyCocoTopDownDataset(data_source_cfg, pipeline)
        ann_info = dataset.data_source.ann_info

        self.assertEqual(len(dataset), 12)
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
