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


class DaliTFRecordMultiViewDatasetTest(unittest.TestCase):

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
        random_area = (0.14, 1.0)
        train_pipeline1 = [
            dict(type='DaliImageDecoder'),
            dict(
                type='DaliRandomResizedCrop',
                size=size,
                random_area=random_area),
            dict(
                type='DaliCropMirrorNormalize',
                crop=[size, size],
                mean=mean,
                std=std,
                crop_pos_x=random.random(),
                crop_pos_y=random.random(),
                prob=0.5)
        ]
        size = 96
        random_area = (0.05, 0.14)
        train_pipeline2 = [
            dict(type='DaliImageDecoder'),
            dict(
                type='DaliRandomResizedCrop',
                size=size,
                random_area=random_area),
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
                type='DaliTFRecordMultiViewDataset',
                data_source=dict(
                    type='ClsSourceImageNetTFRecord',
                    list_file=os.path.join(base_data_root, 'meta/train.txt'),
                    cache_path=target_path),
                num_views=[2, 6],
                pipelines=[train_pipeline1, train_pipeline2],
                distributed=False,
                batch_size=2))
        data = data['train']
        data_loader = build_dataset(data).get_dataloader()
        self.assertEqual(len(io.listdir(target_path)), 6)
        for _, data_batch in enumerate(data_loader):
            self.assertEqual(len(data_batch['img']), 8),
            self.assertEqual(data_batch['img'][0].shape,
                             torch.Size([2, 3, 224, 224]))
            self.assertEqual(data_batch['img'][-1].shape,
                             torch.Size([2, 3, 96, 96]))
            break

        self.assertEqual(len(data_loader), 1877)
        shutil.rmtree(target_path, ignore_errors=True)

    def test_local_train(self):
        base_data_root = SMALL_IMAGENET_TFRECORD_LOCAL
        # dataset settings
        mean = [x * 255 for x in IMG_NORM_CFG['mean']]
        std = [x * 255 for x in IMG_NORM_CFG['std']]
        size = 224
        random_area = (0.14, 1.0)
        train_pipeline1 = [
            dict(type='DaliImageDecoder'),
            dict(
                type='DaliRandomResizedCrop',
                size=size,
                random_area=random_area),
            dict(
                type='DaliCropMirrorNormalize',
                crop=[size, size],
                mean=mean,
                std=std,
                crop_pos_x=random.random(),
                crop_pos_y=random.random(),
                prob=0.5)
        ]
        size = 96
        random_area = (0.05, 0.14)
        train_pipeline2 = [
            dict(type='DaliImageDecoder'),
            dict(
                type='DaliRandomResizedCrop',
                size=size,
                random_area=random_area),
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
                type='DaliTFRecordMultiViewDataset',
                data_source=dict(
                    type='ClsSourceImageNetTFRecord',
                    root=base_data_root,
                    list_file=os.path.join(base_data_root,
                                           'meta/train_relative.txt'),
                ),
                num_views=[2, 6],
                pipelines=[train_pipeline1, train_pipeline2],
                distributed=False,
                batch_size=2))
        data_loader = build_dataset(data['train']).get_dataloader()
        for _, data_batch in enumerate(data_loader):
            self.assertEqual(len(data_batch['img']), 8),
            self.assertEqual(data_batch['img'][0].shape,
                             torch.Size([2, 3, 224, 224]))
            self.assertEqual(data_batch['img'][-1].shape,
                             torch.Size([2, 3, 96, 96]))
            break

        self.assertEqual(len(data_loader), 1877)


if __name__ == '__main__':
    unittest.main()
