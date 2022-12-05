# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import DET_DATASET_PET

from easycv.datasets.builder import build_datasource


class DetSourcePet(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source, cache_at_init, cache_on_the_fly):
        index_list = random.choices(list(range(10)), k=6)
        exclude_list = [i for i in range(7) if i not in index_list]
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
            for i in range(10):
                self.assertIn('img', data_source.samples_list[i])

        if not cache_at_init and cache_on_the_fly:
            for i in index_list:
                self.assertIn('img', data_source.samples_list[i])
            for j in exclude_list:
                self.assertNotIn('img', data_source.samples_list[j])

        if not cache_at_init and not cache_on_the_fly:
            for i in range(10):
                self.assertNotIn('img', data_source.samples_list[i])

        length = len(data_source)
        self.assertEqual(length, 11)

        exists = False
        for idx in range(length):
            result = data_source[idx]
            file_name = result.get('filename', '')
            if file_name.endswith('Abyssinian_110.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (319, 400, 3))
                self.assertEqual(result['gt_labels'].tolist(),
                                 np.array([0], dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].astype(np.int32).tolist(),
                    np.array([[25., 8., 175., 162.]], dtype=np.int32).tolist())
        self.assertTrue(exists)

    def test_default(self):

        cache_at_init = True
        cache_on_the_fly = False
        datasource_cfg = dict(
            type='DetSourcePet',
            path=os.path.join(DET_DATASET_PET, 'test.txt'),
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)
        data_source = build_datasource(datasource_cfg)
        print(data_source[0])
        self._base_test(data_source, cache_at_init, cache_on_the_fly)


if __name__ == '__main__':
    unittest.main()
