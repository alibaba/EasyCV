# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import DET_DATA_SMALL_VOC_LOCAL, VOC_CLASSES

from easycv.datasets.detection.data_sources.voc import DetSourceVOC
from easycv.file import io


class DetSourceVOCTest(unittest.TestCase):

    def setUp(self):
        data_root = DET_DATA_SMALL_VOC_LOCAL
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        for cache_file in io.glob(
                os.path.join(data_root, 'ImageSets/Main/*.cache')):
            io.remove(cache_file)

    def _base_test(self, data_source):
        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source.get_sample(idx)
            self.assertIn('img_shape', data)
            self.assertIn('ori_img_shape', data)
            self.assertIn('filename', data)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['bbox_fields'], ['gt_bboxes'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertGreaterEqual(len(data['gt_labels']), 1)
            self.assertEqual(data['img'].shape[-1], 3)

        length = data_source.get_length()
        self.assertEqual(length, 20)

        exists = False
        for idx in range(length):
            result = data_source.get_sample(idx)
            file_name = result.get('filename', '')
            if file_name.endswith('000032.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (281, 500, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([0, 0, 14, 14], dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].astype(np.int32).tolist(),
                    np.array(
                        [[104., 78., 375., 183.], [133., 88., 197., 123.],
                         [195., 180., 213., 229.], [26., 189., 44., 238.]],
                        dtype=np.int32).tolist())
        self.assertTrue(exists)

    def test_default(self):
        data_root = DET_DATA_SMALL_VOC_LOCAL
        data_source = DetSourceVOC(
            path=os.path.join(data_root, 'ImageSets/Main/train_20.txt'),
            classes=VOC_CLASSES)
        self._base_test(data_source)

    def test_cache_on_the_fly(self):
        data_root = DET_DATA_SMALL_VOC_LOCAL
        data_source = DetSourceVOC(
            path=os.path.join(data_root, 'ImageSets/Main/train_20.txt'),
            classes=VOC_CLASSES,
            cache_at_init=True,
            cache_on_the_fly=False)
        self._base_test(data_source)

    def test_cache_at_init(self):
        data_root = DET_DATA_SMALL_VOC_LOCAL
        data_source = DetSourceVOC(
            path=os.path.join(data_root, 'ImageSets/Main/train_20.txt'),
            classes=VOC_CLASSES,
            cache_at_init=False,
            cache_on_the_fly=True)
        self._base_test(data_source)

    def test_image_root_and_label_root(self):
        data_root = DET_DATA_SMALL_VOC_LOCAL
        data_source = DetSourceVOC(
            path=os.path.join(data_root, 'ImageSets/Main/train_20.txt'),
            classes=VOC_CLASSES,
            img_root_path=os.path.join(data_root, 'JPEGImages'),
            label_root_path=os.path.join(data_root, 'Annotations'),
            cache_at_init=False,
            cache_on_the_fly=True)
        self._base_test(data_source)

    def test_max_retry_num(self):
        data_root = DET_DATA_SMALL_VOC_LOCAL
        data_source = DetSourceVOC(
            path=os.path.join(data_root, 'ImageSets/Main/train_20.txt'),
            classes=VOC_CLASSES,
            img_root_path=os.path.join(data_root, 'fault_path'),
            label_root_path=os.path.join(data_root, 'Annotations'))
        data_source._max_retry_num = 2
        num_samples = data_source.num_samples
        with self.assertRaises(ValueError) as cm:
            for idx in range(num_samples - 1, num_samples * 2):
                _ = data_source.get_sample(idx)

        exception = cm.exception

        self.assertEqual(num_samples, 20)
        self.assertEqual(data_source._retry_count, 2)
        self.assertEqual(exception.args[0], 'All samples failed to load!')


if __name__ == '__main__':
    unittest.main()
