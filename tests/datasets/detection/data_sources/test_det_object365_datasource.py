# Copyright (c) Alibaba, Inc. and its affiliates.

import random
import unittest

import numpy as np
from tests.ut_config import DET_DATASET_OBJECT365

from easycv.datasets.builder import build_datasource


class DetSourceObject365(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source):
        index_list = random.choices(list(range(20)), k=3)

        for idx in index_list:
            data = data_source[idx]
            self.assertIn('ann_info', data)
            self.assertIn('img_info', data)
            self.assertIn('filename', data)
            self.assertEqual(data['img'].shape[-1], 3)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['gt_bboxes'].shape[-1], 4)
            self.assertGreater(len(data['gt_labels']), 1)

        length = len(data_source)

        self.assertEqual(length, 20)

        exists = False
        for idx in range(length):

            result = data_source[idx]

            file_name = result.get('filename', '')

            if file_name.endswith('objects365_v1_00023118.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (512, 768, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([120, 120, 13, 13, 120, 124],
                             dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].tolist(),
                    np.array([[281.78857, 375.06097, 287.3678, 385.66162],
                              [397.64868, 387.58948, 403.33203, 395.81213],
                              [342.46362, 474.97168, 348.0475, 486.62085],
                              [359.11902, 479.59283, 367.7837, 490.66437],
                              [431.1339, 457.85065, 442.27417, 475.9934],
                              [322.3026, 434.84595, 346.9474, 475.31512]],
                             dtype=np.float32).tolist())

        self.assertTrue(exists)

    def test_object365(self):

        data_source = build_datasource(
            dict(
                type='DetSourceObjects365',
                ann_file=DET_DATASET_OBJECT365 + '/val.json',
                img_prefix=DET_DATASET_OBJECT365 + '/images',
                pipeline=[
                    dict(type='LoadImageFromFile', to_float32=True),
                    dict(type='LoadAnnotations', with_bbox=True)
                ],
                filter_empty_gt=False,
                iscrowd=False))

        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
