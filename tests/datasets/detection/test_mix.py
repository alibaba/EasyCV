# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import time
import unittest

import torch
from tests.ut_config import (COCO_CLASSES, DET_DATA_SMALL_COCO_LOCAL,
                             IMG_NORM_CFG_255)

from easycv.datasets.builder import build_dataset


class DetImagesMixDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_load_train(self):
        img_scale = (640, 640)
        scale_ratio = (0.1, 2)
        train_pipeline = [
            dict(type='MMMosaic', img_scale=img_scale, pad_val=114.0),
            dict(
                type='MMRandomAffine',
                scaling_ratio_range=scale_ratio,
                border=(-img_scale[0] // 2, -img_scale[1] // 2)),
            dict(
                type='MMMixUp',  # s m x l; tiny nano will detele
                img_scale=img_scale,
                ratio_range=(0.8, 1.6),
                pad_val=114.0),
            dict(
                type='MMPhotoMetricDistortion',
                brightness_delta=32,
                contrast_range=(0.5, 1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18),
            dict(type='MMRandomFlip', flip_ratio=0.5),
            dict(type='MMResize', keep_ratio=True),
            dict(
                type='MMPad',
                pad_to_square=True,
                pad_val=(114.0, 114.0, 114.0)),
            dict(type='MMNormalize', **IMG_NORM_CFG_255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
        train_dataset = dict(
            type='DetImagesMixDataset',
            data_source=dict(
                type='DetSourceCoco',
                ann_file=os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                                      'instances_train2017_20.json'),
                img_prefix=os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                                        'train2017'),
                pipeline=[
                    dict(type='LoadImageFromFile', to_float32=True),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                classes=COCO_CLASSES,
                filter_empty_gt=False,
                iscrowd=False),
            pipeline=train_pipeline,
            dynamic_scale=img_scale)

        dataset = build_dataset(train_dataset)
        data_num = len(dataset)
        s = time.time()
        for data in dataset:
            pass
        t = time.time()
        print(f'read data done {(t-s)/data_num}s per sample')

        self.assertEqual(data['img'].data.shape, torch.Size([3, 640, 640]))
        img_metas = data['img_metas'].data
        self.assertIn('flip', img_metas)
        self.assertIn('filename', img_metas)
        self.assertIn('img_shape', img_metas)
        self.assertIn('ori_img_shape', img_metas)
        self.assertEqual(data['gt_bboxes'].shape, torch.Size([120, 4]))
        self.assertEqual(data['gt_labels'].shape, torch.Size([120, 1]))
        self.assertEqual(data_num, 20)

    def test_load_test(self):
        img_scale = (640, 640)
        test_pipeline = [
            dict(type='MMResize', img_scale=img_scale, keep_ratio=True),
            dict(
                type='MMPad',
                pad_to_square=True,
                pad_val=(114.0, 114.0, 114.0)),
            dict(type='MMNormalize', **IMG_NORM_CFG_255),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
        val_dataset = dict(
            type='DetImagesMixDataset',
            data_source=dict(
                type='DetSourceCoco',
                ann_file=os.path.join(DET_DATA_SMALL_COCO_LOCAL,
                                      'instances_val2017_20.json'),
                img_prefix=os.path.join(DET_DATA_SMALL_COCO_LOCAL, 'val2017'),
                pipeline=[
                    dict(type='LoadImageFromFile', to_float32=True),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                classes=COCO_CLASSES,
                filter_empty_gt=False,
                iscrowd=True),
            pipeline=test_pipeline,
            dynamic_scale=None,
            label_padding=False)

        dataset = build_dataset(val_dataset)
        data_num = len(dataset)
        s = time.time()
        for data in dataset:
            pass
        t = time.time()
        print(f'read data done {(t-s)/data_num}s per sample')

        self.assertEqual(data['img'].data.shape, torch.Size([3, 640, 640]))
        img_metas = data['img_metas'].data
        self.assertIn('filename', img_metas)
        self.assertIn('img_shape', img_metas)
        self.assertIn('ori_img_shape', img_metas)
        self.assertEqual(data['gt_bboxes'].data.shape, torch.Size([16, 4]))
        self.assertEqual(data['gt_labels'].data.shape, torch.Size([16]))
        self.assertEqual(data_num, 20)


if __name__ == '__main__':
    unittest.main()
