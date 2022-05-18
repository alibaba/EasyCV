# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import DET_DATA_RAW_LOCAL

from easycv.datasets.detection.data_sources.raw import DetSourceRaw
from easycv.file import io


class DetSourceRawTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        data_root = DET_DATA_RAW_LOCAL
        for cache_file in io.glob(os.path.join(data_root, 'labels/*.cache')):
            io.remove(cache_file)

    def _base_test(self, data_source):
        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source.get_sample(idx)
            self.assertIn('img_shape', data)
            self.assertIn('ori_img_shape', data)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['bbox_fields'], ['gt_bboxes'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertGreaterEqual(len(data['gt_labels']), 1)
            self.assertIn('filename', data)
            self.assertEqual(data['img'].shape[-1], 3)

        exclude_idx = [i for i in list(range(20)) if i not in index_list]
        for i in range(3):

            if data_source.cache_at_init:
                self.assertIn('img', data_source.samples_list[exclude_idx[i]])
            if data_source.cache_on_the_fly:
                self.assertNotIn('img',
                                 data_source.samples_list[exclude_idx[i]])

        length = data_source.get_length()
        self.assertEqual(length, 126)

        exists = False
        for idx in range(length):
            result = data_source.get_sample(idx)
            file_name = result.get('filename', '')
            if file_name.endswith('000000000086.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (640, 512, 3))
                self.assertEqual(result['gt_labels'].tolist(),
                                 np.array([0, 3, 26], dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].astype(np.int32).tolist(),
                    np.array([[152.11008, 183.09982, 281.26004, 559.0697],
                              [130.13991, 346.51968, 424.40015, 635.15967],
                              [265.10004, 294.5901, 399.7002, 404.7699]],
                             dtype=np.int32).tolist())
        self.assertTrue(exists)

    def test_default(self):
        data_root = DET_DATA_RAW_LOCAL
        data_source = DetSourceRaw(
            img_root_path=os.path.join(data_root, 'images/train2017'),
            label_root_path=os.path.join(data_root, 'labels/train2017'),
        )
        self._base_test(data_source)

    def test_cache_on_the_fly(self):
        data_root = DET_DATA_RAW_LOCAL
        data_source = DetSourceRaw(
            img_root_path=os.path.join(data_root, 'images/train2017'),
            label_root_path=os.path.join(data_root, 'labels/train2017'),
            cache_on_the_fly=True,
            cache_at_init=False,
        )
        self._base_test(data_source)

    def test_cache_at_init(self):
        data_root = DET_DATA_RAW_LOCAL
        data_source = DetSourceRaw(
            img_root_path=os.path.join(data_root, 'images/train2017'),
            label_root_path=os.path.join(data_root, 'labels/train2017'),
            cache_on_the_fly=False,
            cache_at_init=True,
        )
        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
