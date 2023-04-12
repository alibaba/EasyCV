# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import unittest

import numpy as np
from tests.ut_config import (COCO_STUFF_CLASSES,
                             SEG_DATA_SAMLL_COCO_STUFF_164K,
                             SEG_DATA_SMALL_COCO_STUFF_10K)

from easycv.datasets.builder import build_datasource


class SegSourceCocoStuffTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source, cache_at_init, cache_on_the_fly, num):
        index_list = random.choices(list(range(num)), k=3)
        for idx in index_list:
            data = data_source[idx]
            self.assertIn('filename', data)
            self.assertIn('seg_filename', data)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['seg_fields'], ['gt_semantic_seg'])
            self.assertIn('img_shape', data)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['gt_semantic_seg'].shape,
                             data['img_shape'][:2])
            self.assertEqual(data['img'].shape[-1], 3)

            self.assertTrue(
                len(np.unique(data['gt_semantic_seg'])) < len(
                    COCO_STUFF_CLASSES))

        exclude_idx = [i for i in list(range(num)) if i not in index_list]

        if cache_at_init:
            for i in range(num):
                self.assertIn('img', data_source.samples_list[i])

        if not cache_at_init and cache_on_the_fly:
            for i in index_list:
                self.assertIn('img', data_source.samples_list[i])
            for j in exclude_idx:
                self.assertNotIn('img', data_source.samples_list[j])

        if not cache_at_init and not cache_on_the_fly:
            for i in range(num):
                print(data_source.samples_list[i])
                self.assertNotIn('img', data_source.samples_list[i])

        length = len(data_source)
        self.assertEqual(length, num)
        self.assertEqual(data_source.PALETTE.shape,
                         (len(COCO_STUFF_CLASSES), 3))

    def test_cocostuff10k(self):
        data_root = SEG_DATA_SMALL_COCO_STUFF_10K
        cache_at_init = True
        cache_on_the_fly = False
        data_source = build_datasource(
            dict(
                type='SegSourceCocoStuff10k',
                path=os.path.join(data_root, 'all.txt'),
                img_root=os.path.join(data_root, 'images'),
                label_root=os.path.join(data_root, 'lable'),
                cache_at_init=cache_at_init,
                cache_on_the_fly=cache_on_the_fly,
                classes=COCO_STUFF_CLASSES))

        self._base_test(data_source, cache_at_init, cache_on_the_fly, 10)

        exists = False
        for idx in range(len(data_source)):
            result = data_source[idx]
            file_name = result.get('filename', '')
            if file_name.endswith('COCO_train2014_000000000349.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (480, 640, 3))
                self.assertEqual(
                    np.unique(result['gt_semantic_seg']).tolist(),
                    [0, 7, 64, 95, 96, 106, 126, 169])
        self.assertTrue(exists)

    def test_cocostuff164k(self):
        data_root = SEG_DATA_SAMLL_COCO_STUFF_164K
        cache_at_init = True
        cache_on_the_fly = False
        data_source = build_datasource(
            dict(
                type='SegSourceCocoStuff164k',
                img_root=os.path.join(data_root, 'images'),
                label_root=os.path.join(data_root, 'label'),
                cache_at_init=cache_at_init,
                cache_on_the_fly=cache_on_the_fly,
                classes=COCO_STUFF_CLASSES))

        self._base_test(data_source, cache_at_init, cache_on_the_fly,
                        len(data_source))

        exists = False
        for idx in range(len(data_source)):
            result = data_source[idx]
            file_name = result.get('filename', '')
            if file_name.endswith('000000000009.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (480, 640, 3))
                self.assertEqual(
                    np.unique(result['gt_semantic_seg']).tolist(),
                    [50, 54, 55, 120, 142, 164, 255])
        self.assertTrue(exists)


if __name__ == '__main__':
    unittest.main()
