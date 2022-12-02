# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import random
import unittest

import numpy as np
from tests.ut_config import DET_DATASET_CROWD_HUMAN

from easycv.datasets.builder import build_datasource


class DetSourceArtaxorTest(unittest.TestCase):

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
        self.assertEqual(length, 12)

        exists = False
        for idx in range(length):
            result = data_source[idx]
            file_name = result.get('filename', '')
            if file_name.endswith('273271,1acb00092ad10cd.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (494, 692, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                             dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].astype(np.int32).tolist(),
                    np.array(
                        [[61., 242., 267., 494.], [313., 97., 453., 429.],
                         [461., 230., 565., 433.], [373., 247., 471., 407.],
                         [297., 202., 397., 433.], [217., 69., 294., 428.],
                         [208., 226., 316., 413.], [120., 44., 216., 343.],
                         [481., 42., 539., 113.], [0., 21., 60., 95.],
                         [125., 24., 166., 101.], [234., 29., 269., 96.],
                         [584., 43., 649., 112.]],
                        dtype=np.int32).tolist())
        self.assertTrue(exists)

    def test_default(self):

        cache_at_init = True
        cache_on_the_fly = False

        datasource_cfg = dict(
            type='DetSourceCrowdHuman',
            ann_file=DET_DATASET_CROWD_HUMAN + '/train.odgt',
            img_prefix=DET_DATASET_CROWD_HUMAN + '/Images',
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)

        data_source = build_datasource(datasource_cfg)
        self._base_test(data_source, cache_at_init, cache_on_the_fly)


if __name__ == '__main__':
    unittest.main()
