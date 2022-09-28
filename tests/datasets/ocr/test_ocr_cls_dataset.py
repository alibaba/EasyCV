# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from tests.ut_config import SMALL_OCR_CLS_DATA

from easycv.datasets.builder import build_dataset


class OCRClsDatasetTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _get_dataset(self):
        data_root = SMALL_OCR_CLS_DATA
        data_train_list = os.path.join(data_root, 'label.txt')
        pipeline = [
            dict(type='RecAug', use_tia=False),
            dict(type='ClsResizeImg', img_shape=(3, 48, 192)),
            dict(type='MMToTensor'),
            dict(
                type='Collect', keys=['img', 'label'], meta_keys=['img_path'])
        ]
        data = dict(
            train=dict(
                type='OCRClsDataset',
                data_source=dict(
                    type='OCRClsSource',
                    label_file=data_train_list,
                    data_dir=SMALL_OCR_CLS_DATA + '/img',
                    label_list=['0', '180'],
                ),
                pipeline=pipeline))
        dataset = build_dataset(data['train'])

        return dataset

    def test_default(self):
        dataset = self._get_dataset()
        for _, batch in enumerate(dataset):
            img, target = batch['img'], batch['label']
            self.assertEqual(img.shape, torch.Size([3, 48, 192]))
            self.assertIn(target, list(range(2)))
            break


if __name__ == '__main__':
    unittest.main()
