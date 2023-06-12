# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import COCO_CLASSES, DET_DATA_MANIFEST_OSS

from easycv.datasets.detection.data_sources.pai_format import DetSourcePAI
from easycv.file import io


class DetSourcePAITest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _test_base(self, data_source, cache_at_init, cache_on_the_fly):
        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source[idx]
            self.assertIn('img_shape', data)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['bbox_fields'], ['gt_bboxes'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertGreater(len(data['gt_labels']), 1)
            self.assertIn('filename', data)
            self.assertEqual(data['img'].shape[-1], 3)

        exclude_idx = [i for i in list(range(20)) if i not in index_list]
        if cache_at_init:
            for i in range(20):
                self.assertIn('img', data_source.samples_list[i])

        if not cache_at_init and cache_on_the_fly:
            for i in index_list:
                self.assertIn('img', data_source.samples_list[i])
            for j in exclude_idx:
                self.assertNotIn('img', data_source.samples_list[j])

        if not cache_at_init and not cache_on_the_fly:
            for i in range(20):
                self.assertNotIn('img', data_source.samples_list[i])

        length = len(data_source)
        self.assertEqual(length, 20)

        exists = False
        for idx in range(length):
            file_name = data_source.samples_list[idx]['filename']
            if file_name.endswith('000000060623.jpg'):
                exists = True
                data = data_source[idx]
                self.assertEqual(
                    data['gt_labels'].tolist(),
                    np.array([0, 0, 44, 45, 40, 60, 0],
                             dtype=np.int32).tolist())
                self.assertEqual(
                    data['gt_bboxes'].astype(np.int32).tolist(),
                    np.array([[
                        1.9100037e+00, 1.9100037e+00, 3.4754001e+02,
                        4.2220999e+02
                    ],
                              [
                                  2.8338000e+02, 2.1700134e+00, 5.7737000e+02,
                                  3.2700000e+02
                              ],
                              [
                                  4.1187003e+02, 1.2767001e+02, 5.3591003e+02,
                                  2.1360001e+02
                              ],
                              [
                                  4.1354001e+02, 2.3705998e+02, 5.4337000e+02,
                                  3.5246997e+02
                              ],
                              [
                                  5.6067004e+02, 3.5849998e+01, 6.4000000e+02,
                                  2.0903000e+02
                              ],
                              [
                                  3.4314001e+02, 1.0602998e+02, 6.4000000e+02,
                                  4.2700000e+02
                              ],
                              [
                                  4.7057996e+02, 5.5999947e-01, 6.2072998e+02,
                                  4.9029999e+01
                              ]],
                             dtype=np.int32).tolist())

        self.assertTrue(exists)

    def test_default(self):
        io.access_oss()
        data_root = DET_DATA_MANIFEST_OSS
        cache_at_init = False
        cache_on_the_fly = False
        data_source = DetSourcePAI(
            path=os.path.join(data_root, 'train2017_20.manifest'),
            classes=COCO_CLASSES,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)

        self._test_base(data_source, cache_at_init, cache_on_the_fly)

    def test_cache_on_the_fly(self):
        io.access_oss()
        data_root = DET_DATA_MANIFEST_OSS
        cache_on_the_fly = True
        cache_at_init = False
        data_source = DetSourcePAI(
            path=os.path.join(data_root, 'train2017_20.manifest'),
            classes=COCO_CLASSES,
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)

        self._test_base(data_source, cache_at_init, cache_on_the_fly)

    def test_cache_at_init(self):
        io.access_oss()
        data_root = DET_DATA_MANIFEST_OSS
        cache_on_the_fly = False
        cache_at_init = True
        data_source = DetSourcePAI(
            path=os.path.join(data_root, 'train2017_20.manifest'),
            classes=COCO_CLASSES,
            cache_on_the_fly=cache_on_the_fly,
            cache_at_init=cache_at_init)

        self._test_base(data_source, cache_at_init, cache_on_the_fly)


if __name__ == '__main__':
    unittest.main()
