# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import shutil
import unittest
import uuid

import torch
from tests.ut_config import (IMG_NORM_CFG, SMALL_IMAGENET_TFRECORD_LOCAL,
                             SMALL_IMAGENET_TFRECORD_OSS, TMP_DIR_LOCAL)

from easycv.datasets import build_dataset
from easycv.file import io


class DaliImageNetTFRecordDataSetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_oss_train(self):
        io.access_oss()
        base_data_root = SMALL_IMAGENET_TFRECORD_OSS
        target_path = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        # dataset settings
        mean = [x * 255 for x in IMG_NORM_CFG['mean']]
        std = [x * 255 for x in IMG_NORM_CFG['std']]
        size = 224
        train_pipeline = [
            dict(type='DaliImageDecoder'),
            dict(
                type='DaliRandomResizedCrop',
                size=size,
                random_area=(0.08, 1.0)),
            dict(
                type='DaliCropMirrorNormalize',
                crop=[size, size],
                mean=mean,
                std=std,
                crop_pos_x=random.random(),
                crop_pos_y=random.random(),
                prob=0.5)
        ]
        data = dict(
            imgs_per_gpu=2,
            workers_per_gpu=2,
            train=dict(
                type='DaliImageNetTFRecordDataSet',
                data_source=dict(
                    type='ClsSourceImageNetTFRecord',
                    list_file=os.path.join(base_data_root, 'meta/train.txt'),
                    cache_path=target_path),
                pipeline=train_pipeline,
                distributed=False,
                batch_size=2,
                label_offset=1))
        data = data['train']
        data_loader = build_dataset(data).get_dataloader()
        self.assertEqual(len(io.listdir(target_path)), 6)
        self.assertEqual(len(data_loader), 1877)
        for _, data_batch in enumerate(data_loader):
            self.assertEqual(data_batch['img'].shape,
                             torch.Size([2, 3, 224, 224]))
            self.assertEqual(data_batch['gt_label'].shape, torch.Size([2]))
            labels = data_batch['gt_label'].cpu().numpy()
            for l in labels:
                self.assertTrue(l in range(1000))
            break

        shutil.rmtree(target_path, ignore_errors=True)

    def test_local_val(self):
        base_data_root = SMALL_IMAGENET_TFRECORD_LOCAL
        # dataset settings
        mean = [x * 255 for x in IMG_NORM_CFG['mean']]
        std = [x * 255 for x in IMG_NORM_CFG['std']]
        size = 224
        val_pipeline = [
            dict(type='DaliImageDecoder'),
            dict(type='DaliResize', resize_shorter=size * 1.15),
            dict(
                type='DaliCropMirrorNormalize',
                crop=[size, size],
                mean=mean,
                std=std,
                prob=0.0)
        ]
        data = dict(
            imgs_per_gpu=2,
            workers_per_gpu=2,
            val=dict(
                type='DaliImageNetTFRecordDataSet',
                data_source=dict(
                    type='ClsSourceImageNetTFRecord',
                    root=base_data_root,
                    list_file=os.path.join(base_data_root,
                                           'meta/train_relative.txt'),
                ),
                pipeline=val_pipeline,
                random_shuffle=False,
                distributed=False,
                batch_size=2,
                label_offset=1))
        data = data['val']
        data_loader = build_dataset(data).get_dataloader()
        self.assertEqual(len(data_loader), 1877)
        for _, data_batch in enumerate(data_loader):
            self.assertEqual(data_batch['img'].shape,
                             torch.Size([2, 3, 224, 224]))
            self.assertEqual(data_batch['gt_label'].shape, torch.Size([2]))
            labels = data_batch['gt_label'].cpu().numpy()
            for l in labels:
                self.assertTrue(l in range(1000))
            break


if __name__ == '__main__':
    unittest.main()
