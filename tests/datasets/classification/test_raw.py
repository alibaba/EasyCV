# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from tests.ut_config import IMG_NORM_CFG, SMALL_IMAGENET_RAW_LOCAL

from easycv.datasets.builder import build_dataset


class ClsDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        data_root = SMALL_IMAGENET_RAW_LOCAL
        data_train_list = os.path.join(data_root, 'meta/train_labeled_200.txt')
        pipeline = [
            dict(type='RandomResizedCrop', size=224),
            dict(type='RandomHorizontalFlip'),
            dict(type='ToTensor'),
            dict(type='Normalize', **IMG_NORM_CFG),
            dict(type='Collect', keys=['img', 'gt_labels'])
        ]
        data = dict(
            train=dict(
                type='ClsDataset',
                data_source=dict(
                    list_file=data_train_list,
                    root=os.path.join(data_root, 'train'),
                    type='ClsSourceImageList'),
                pipeline=pipeline))
        dataset = build_dataset(data['train'])

        for _, batch in enumerate(dataset):
            img, target = batch['img'], batch['gt_labels']
            self.assertEqual(img.shape, torch.Size([3, 224, 224]))
            self.assertIn(target, list(range(1000)))
            break


if __name__ == '__main__':
    unittest.main()
