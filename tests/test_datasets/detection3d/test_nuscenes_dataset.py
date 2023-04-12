# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from tests.ut_config import NUSCENES_CLASSES, SMALL_NUSCENES_PATH

from easycv.datasets.detection3d import NuScenesDataset


class NuScenesDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _get_dataset(self):
        meta_keys = ('filename', 'ori_shape', 'img_shape', 'lidar2img',
                     'depth2img', 'cam2img', 'pad_shape', 'scale_factor',
                     'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip',
                     'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 'pcd_trans',
                     'sample_idx', 'prev_idx', 'next_idx', 'pcd_scale_factor',
                     'pcd_rotation', 'pts_filename', 'transformation_3d_flow',
                     'scene_token', 'can_bus')
        point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        img_norm_cfg = dict(
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True)
        input_modality = dict(
            use_lidar=False,
            use_camera=True,
            use_radar=False,
            use_map=False,
            use_external=True)
        data_source = dict(
            type='Det3dSourceNuScenes',
            data_root=SMALL_NUSCENES_PATH,
            ann_file=os.path.join(SMALL_NUSCENES_PATH,
                                  'nuscenes_infos_temporal_train_20.pkl'),
            pipeline=[
                dict(type='LoadMultiViewImageFromFiles', to_float32=True),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True,
                    with_attr_label=False)
            ],
            classes=NUSCENES_CLASSES,
            modality=input_modality,
            test_mode=False,
            use_valid_flag=True,
            box_type_3d='LiDAR')
        train_pipeline = [
            dict(type='PhotoMetricDistortionMultiViewImage'),
            dict(
                type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='ObjectNameFilter', classes=NUSCENES_CLASSES),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(type='DefaultFormatBundle3D', class_names=NUSCENES_CLASSES),
            dict(
                type='Collect3D',
                keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                meta_keys=meta_keys)
        ]

        dataset = NuScenesDataset(
            data_source=data_source, pipeline=train_pipeline, queue_length=3)

        return dataset

    def test_load(self):
        dataset = self._get_dataset()
        self.assertEqual(len(dataset), 20)

        for i, data in enumerate(dataset):
            if i > 2:
                break
            if i == 0:
                shape_num = 81
            elif i == 1:
                shape_num = 93
            elif i == 2:
                shape_num = 109
            self.assertEqual(data['img']._data.shape,
                             torch.Size([3, 6, 3, 480, 800]))
            self.assertEqual(data['gt_labels_3d']._data.shape,
                             torch.Size([shape_num]))
            self.assertEqual(data['gt_bboxes_3d']._data.bev.shape,
                             torch.Size([shape_num, 5]))
            self.assertEqual(data['gt_bboxes_3d']._data.bottom_center.shape,
                             torch.Size([shape_num, 3]))
            self.assertEqual(data['gt_bboxes_3d']._data.bottom_height.shape,
                             torch.Size([shape_num]))
            self.assertEqual(data['gt_bboxes_3d']._data.center.shape,
                             torch.Size([shape_num, 3]))
            self.assertEqual(data['gt_bboxes_3d']._data.volume.shape,
                             torch.Size([shape_num]))
            self.assertEqual(data['gt_bboxes_3d']._data.yaw.shape,
                             torch.Size([shape_num]))
            img_metas = data['img_metas'].data
            self.assertEqual(len(img_metas), 3)
            for _, img_meta in img_metas.items():
                self.assertTrue(len(img_meta['filename']), 6)
                self.assertTrue(len(img_meta['img_shape']), 6)
                self.assertTrue(len(img_meta['ori_shape']), 6)
                self.assertTrue(len(img_meta['lidar2img']), 6)
                self.assertTrue(len(img_meta['pad_shape']), 6)
                self.assertIn('can_bus', img_meta)
                self.assertIn('scene_token', img_meta)
                self.assertIn('scale_factor', img_meta)
                self.assertIn('box_mode_3d', img_meta)
                self.assertIn('sample_idx', img_meta)
                self.assertIn('prev_idx', img_meta)
                self.assertIn('next_idx', img_meta)
                self.assertIn('pts_filename', img_meta)


if __name__ == '__main__':
    unittest.main()
