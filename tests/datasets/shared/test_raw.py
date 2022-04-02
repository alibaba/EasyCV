# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from tests.ut_config import IMG_NORM_CFG, SMALL_IMAGENET_RAW_LOCAL

from easycv.datasets.builder import build_dataset


class RawDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_raw_dataset_with_label(self):
        data_train_list = os.path.join(SMALL_IMAGENET_RAW_LOCAL,
                                       'meta/train_labeled_200.txt')
        data_train_root = os.path.join(SMALL_IMAGENET_RAW_LOCAL, 'train')
        train_data = dict(
            type='RawDataset',
            with_label=True,
            data_source=dict(
                type='ClsSourceImageList',
                list_file=data_train_list,
                root=data_train_root),
            pipeline=[
                dict(type='Resize', size=256),
                dict(type='CenterCrop', size=224),
                dict(type='ToTensor'),
                dict(type='Normalize', **IMG_NORM_CFG),
            ])

        dataset = build_dataset(train_data)

        for _, batch in enumerate(dataset):
            self.assertEqual(batch['img'].shape, torch.Size([3, 224, 224]))
            self.assertIn(batch['gt_label'], list(range(1000)))
            break

        self.assertEqual(len(dataset), 200)

    def test_raw_dataset(self):
        data_train_list = os.path.join(SMALL_IMAGENET_RAW_LOCAL,
                                       'meta/train_200.txt')
        data_train_root = SMALL_IMAGENET_RAW_LOCAL
        train_data = dict(
            type='RawDataset',
            with_label=False,
            data_source=dict(
                type='SSLSourceImageList',
                list_file=data_train_list,
                root=data_train_root),
            pipeline=[
                dict(type='Resize', size=256),
                dict(type='Resize', size=(224, 224)),
                dict(type='ToTensor'),
                dict(type='Normalize', **IMG_NORM_CFG),
            ])

        dataset = build_dataset(train_data)

        for _, batch in enumerate(dataset):
            self.assertEqual(batch['img'].shape, torch.Size([3, 224, 224]))
            break

        self.assertEqual(len(dataset), 200)


if __name__ == '__main__':
    unittest.main()
