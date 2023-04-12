# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from PIL import Image
from tests.ut_config import IMG_NORM_CFG, SMALL_IMAGENET_RAW_LOCAL

from easycv.datasets.builder import build_dataset


class MultiViewDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_multiview_dataset(self):
        data_train_list = os.path.join(SMALL_IMAGENET_RAW_LOCAL,
                                       'meta/train_200.txt')
        data_train_root = SMALL_IMAGENET_RAW_LOCAL
        pipeline1 = [
            dict(
                type='RandomResizedCrop',
                size=224,
                scale=(0.4, 1.),
                interpolation=Image.BICUBIC),
            dict(type='RandomHorizontalFlip', p=0.5),
            dict(
                type='RandomAppliedTrans',
                transforms=[
                    dict(
                        type='ColorJitter',
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1)
                ],
                p=0.8),
            dict(type='RandomGrayscale', p=0.2),
            dict(
                type='RandomAppliedTrans',
                transforms=[dict(type='GaussianBlur', kernel_size=23)],
                p=1.0),
            dict(type='ToTensor'),
            dict(type='Normalize', **IMG_NORM_CFG),
            dict(type='Collect', keys=['img'])
        ]
        pipeline2 = [
            dict(
                type='RandomResizedCrop',
                size=224,
                scale=(0.4, 1.),
                interpolation=Image.BICUBIC),
            dict(type='RandomHorizontalFlip', p=0.5),
            dict(
                type='RandomAppliedTrans',
                transforms=[
                    dict(
                        type='ColorJitter',
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1)
                ],
                p=0.8),
            dict(type='RandomGrayscale', p=0.2),
            dict(
                type='RandomAppliedTrans',
                transforms=[dict(type='GaussianBlur', kernel_size=23)],
                p=0.1),
            dict(
                type='RandomAppliedTrans',
                transforms=[dict(type='Solarization', threshold=130)],
                p=0.2),
            dict(type='ToTensor'),
            dict(type='Normalize', **IMG_NORM_CFG),
            dict(type='Collect', keys=['img'])
        ]
        pipeline3 = [
            dict(
                type='RandomResizedCrop',
                size=96,
                scale=(0.05, 0.4),
                interpolation=Image.BICUBIC),
            dict(type='RandomHorizontalFlip', p=0.5),
            dict(
                type='RandomAppliedTrans',
                transforms=[
                    dict(
                        type='ColorJitter',
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.1)
                ],
                p=0.8),
            dict(type='RandomGrayscale', p=0.2),
            dict(
                type='RandomAppliedTrans',
                transforms=[dict(type='GaussianBlur', kernel_size=23)],
                p=0.5),
            dict(type='ToTensor'),
            dict(type='Normalize', **IMG_NORM_CFG),
            dict(type='Collect', keys=['img'])
        ]

        train_data = dict(
            type='MultiViewDataset',
            data_source=dict(
                type='SSLSourceImageList',
                list_file=data_train_list,
                root=data_train_root),
            num_views=[1, 1, 8],
            pipelines=[pipeline1, pipeline2, pipeline3])

        dataset = build_dataset(train_data)
        for _, batch in enumerate(dataset):
            self.assertEqual(len(batch['img']), 10)
            self.assertEqual(batch['img'][0].shape, torch.Size([3, 224, 224]))
            self.assertEqual(batch['img'][-1].shape, torch.Size([3, 96, 96]))
            break


if __name__ == '__main__':
    unittest.main()
