# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from tests.ut_config import (IMG_NORM_CFG, SMALL_OCR_DET_DATA,
                             SMALL_OCR_DET_PAI_DATA)

from easycv.datasets.builder import build_dataset


class OCRDetDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _get_dataset(self):
        data_root = SMALL_OCR_DET_DATA
        data_train_list = os.path.join(data_root, 'label.txt')
        pipeline = [
            dict(
                type='IaaAugment',
                augmenter_args=[{
                    'type': 'Fliplr',
                    'args': {
                        'p': 0.5
                    }
                }, {
                    'type': 'Affine',
                    'args': {
                        'rotate': [-10, 10]
                    }
                }, {
                    'type': 'Resize',
                    'args': {
                        'size': [0.5, 3]
                    }
                }]),
            dict(
                type='EastRandomCropData',
                size=[640, 640],
                max_tries=50,
                keep_ratio=True),
            dict(
                type='MakeBorderMap',
                shrink_ratio=0.4,
                thresh_min=0.3,
                thresh_max=0.7),
            dict(type='MakeShrinkMap', shrink_ratio=0.4, min_text_size=8),
            dict(type='MMNormalize', **IMG_NORM_CFG),
            dict(
                type='ImageToTensor',
                keys=[
                    'img', 'threshold_map', 'threshold_mask', 'shrink_map',
                    'shrink_mask'
                ]),
            dict(
                type='Collect',
                keys=[
                    'img', 'threshold_map', 'threshold_mask', 'shrink_map',
                    'shrink_mask'
                ]),
        ]
        data = dict(
            train=dict(
                type='OCRDetDataset',
                data_source=dict(
                    type='OCRDetSource',
                    label_file=data_train_list,
                    data_dir=SMALL_OCR_DET_DATA + '/img',
                ),
                pipeline=pipeline))
        dataset = build_dataset(data['train'])

        return dataset

    def _get_dataset_pai(self):
        data_root = SMALL_OCR_DET_PAI_DATA
        data_train_list = os.path.join(data_root, 'label.csv')
        pipeline = [
            dict(
                type='IaaAugment',
                augmenter_args=[{
                    'type': 'Fliplr',
                    'args': {
                        'p': 0.5
                    }
                }, {
                    'type': 'Affine',
                    'args': {
                        'rotate': [-10, 10]
                    }
                }, {
                    'type': 'Resize',
                    'args': {
                        'size': [0.5, 3]
                    }
                }]),
            dict(
                type='EastRandomCropData',
                size=[640, 640],
                max_tries=50,
                keep_ratio=True),
            dict(
                type='MakeBorderMap',
                shrink_ratio=0.4,
                thresh_min=0.3,
                thresh_max=0.7),
            dict(type='MakeShrinkMap', shrink_ratio=0.4, min_text_size=8),
            dict(type='MMNormalize', **IMG_NORM_CFG),
            dict(
                type='ImageToTensor',
                keys=[
                    'img', 'threshold_map', 'threshold_mask', 'shrink_map',
                    'shrink_mask'
                ]),
            dict(
                type='Collect',
                keys=[
                    'img', 'threshold_map', 'threshold_mask', 'shrink_map',
                    'shrink_mask'
                ]),
        ]
        data = dict(
            train=dict(
                type='OCRDetDataset',
                data_source=dict(
                    type='OCRPaiDetSource',
                    label_file=[data_train_list],
                    data_dir=SMALL_OCR_DET_PAI_DATA + '/img',
                ),
                pipeline=pipeline))
        dataset = build_dataset(data['train'])

        return dataset

    def test_default(self):
        dataset = self._get_dataset()
        for _, batch in enumerate(dataset):
            img, threshold_mask, shrink_mask = batch['img'], batch[
                'threshold_mask'], batch['shrink_mask']
            self.assertEqual(img.shape, torch.Size([3, 640, 640]))
            self.assertEqual(threshold_mask.shape, torch.Size([1, 640, 640]))
            self.assertEqual(shrink_mask.shape, torch.Size([1, 640, 640]))
            break

    def test_pai(self):
        dataset = self._get_dataset_pai()
        for _, batch in enumerate(dataset):
            img, threshold_mask, shrink_mask = batch['img'], batch[
                'threshold_mask'], batch['shrink_mask']
            self.assertEqual(img.shape, torch.Size([3, 640, 640]))
            self.assertEqual(threshold_mask.shape, torch.Size([1, 640, 640]))
            self.assertEqual(shrink_mask.shape, torch.Size([1, 640, 640]))
            break


if __name__ == '__main__':
    unittest.main()
