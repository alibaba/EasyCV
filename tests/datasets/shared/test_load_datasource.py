# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import (COCO_CLASSES, DET_DATA_COCO2017_DOWNLOAD,
                             DET_DATA_SMALL_VOC_LOCAL, VOC_CLASSES)

from easycv.datasets.builder import build_datasource, load_datasource
from easycv.file import io


class Load_DataSource(unittest.TestCase):

    def setUp(self):
        data_root = DET_DATA_SMALL_VOC_LOCAL
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        for cache_file in io.glob(
                os.path.join(data_root, 'ImageSets/Main/*.cache')):
            io.remove(cache_file)

    def _base_test_voc(self, data_source, cache_at_init, cache_on_the_fly):
        index_list = random.choices(list(range(20)), k=3)
        exclude_list = [i for i in range(20) if i not in index_list]
        for idx in index_list:
            data = data_source[idx]
            self.assertIn('img_shape', data)
            self.assertIn('ori_img_shape', data)
            self.assertIn('filename', data)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['bbox_fields'], ['gt_bboxes'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertGreaterEqual(len(data['gt_labels']), 1)
            self.assertEqual(data['img'].shape[-1], 3)

        if cache_at_init:
            for i in range(20):
                self.assertIn('img', data_source.samples_list[i])

        if not cache_at_init and cache_on_the_fly:
            for i in index_list:
                self.assertIn('img', data_source.samples_list[i])
            for j in exclude_list:
                self.assertNotIn('img', data_source.samples_list[j])

        if not cache_at_init and not cache_on_the_fly:
            for i in range(20):
                self.assertNotIn('img', data_source.samples_list[i])

        length = len(data_source)
        self.assertEqual(length, 20)

        exists = False
        for idx in range(length):
            result = data_source[idx]
            file_name = result.get('filename', '')
            if file_name.endswith('000032.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (281, 500, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([0, 0, 14, 14], dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].astype(np.int32).tolist(),
                    np.array(
                        [[104., 78., 375., 183.], [133., 88., 197., 123.],
                         [195., 180., 213., 229.], [26., 189., 44., 238.]],
                        dtype=np.int32).tolist())
        self.assertTrue(exists)

    def _base_test_coco_detection(self, data_source):

        index_list = random.choices(list(range(20)), k=3)

        for idx in index_list:
            data = data_source[idx]
            self.assertIn('ann_info', data)
            self.assertIn('img_info', data)
            self.assertIn('filename', data)
            self.assertEqual(data['img'].shape[-1], 3)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertGreater(len(data['gt_labels']), 1)

        length = len(data_source)

        self.assertEqual(length, 20)

        exists = False
        for idx in range(length):

            result = data_source[idx]

            file_name = result.get('filename', '')

            if file_name.endswith('000000224736.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (427, 640, 3))
                self.assertEqual(result['gt_labels'].tolist(),
                                 np.array([61, 71], dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].tolist(),
                    np.array([[148.1, 297.65, 270.24, 383.24],
                              [470.09, 148.13, 552.07, 207.29]],
                             dtype=np.float32).tolist())

        self.assertTrue(exists)

    def _base_test_coco_pose(self, data_source):

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

    def test_voc2007_load_datasource(self):
        data_root = DET_DATA_COCO2017_DOWNLOAD
        cache_at_init = False
        cache_on_the_fly = False

        data_cfg = dict(
            type='DetSourceVOC',
            name='voc2007',
            split='train',
            task='detection',
            target_dir=data_root,
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=VOC_CLASSES,
            filter_empty_gt=True,
            iscrowd=False)

        cfg = load_datasource(data_cfg)

        data_source = build_datasource(cfg)
        self._base_test(
            data_source,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)

    def test_voc2012_load_datasource(self):
        data_root = DET_DATA_COCO2017_DOWNLOAD
        cache_at_init = False
        cache_on_the_fly = False

        data_cfg = dict(
            type='DetSourceVOC',
            name='voc2012',
            split='train',
            task='detection',
            target_dir=data_root,
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=VOC_CLASSES,
            filter_empty_gt=True,
            iscrowd=False)

        cfg = load_datasource(data_cfg)

        data_source = build_datasource(cfg)
        self._base_test_voc(
            data_source,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)

    def test_coco2017_detection_load_datasource(self):
        data_root = DET_DATA_COCO2017_DOWNLOAD
        data_cfg = dict(
            type='DetSourceCoco',
            name='coco2017',
            split='train',
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            target_dir=data_root,
            task='detection',
            classes=COCO_CLASSES,
            filter_empty_gt=True,
            iscrowd=False)

        cfg = load_datasource(data_cfg)

        data_source = build_datasource(cfg)
        self._base_test_coco_detection(data_source)

    def test_coco2017_pose_load_datasource(self):
        data_root = DET_DATA_COCO2017_DOWNLOAD
        channel_cfg = dict(
            num_output_channels=17,
            dataset_joints=17,
            dataset_channel=[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            ],
            inference_channel=[
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
            ])
        data_cfg = dict(
            image_size=[192, 256],
            heatmap_size=[48, 64],
            num_output_channels=channel_cfg['num_output_channels'],
            num_joints=channel_cfg['dataset_joints'],
            dataset_channel=channel_cfg['dataset_channel'],
            inference_channel=channel_cfg['inference_channel'],
            soft_nms=False,
            nms_thr=1.0,
            oks_thr=0.9,
            vis_thr=0.2,
            use_gt_bbox=True,
            det_bbox_thr=0.0,
            bbox_file='',
        )
        data_source_cfg = dict(type='PoseTopDownSourceCoco', data_cfg=data_cfg)
        data_cfg = dict(
            name='coco2017',
            split='train',
            target_dir=data_root,
            task='pose',
            **data_source_cfg)

        cfg = load_datasource(data_cfg)

        data_source = build_datasource(cfg)
        self._base_test_coco_pose(data_source)


if __name__ == '__main__':
    unittest.main()
