# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
from tests.ut_config import DET_DATASET_TINY_PERSON

from easycv.datasets.detection.data_sources.coco import DetSourceTinyPerson


class DetSourceCocoTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source):
        index_list = random.choices(list(range(19)), k=1)

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

        self.assertEqual(length, 19)

        exists = False
        for idx in range(length):

            result = data_source[idx]

            file_name = result.get('filename', '')

            if file_name.endswith('bb_V0005_I0006680.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (1080, 1920, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].tolist(),
                    np.array([[706.20715, 190.13815, 716.2589, 211.50473],
                              [783.45087, 214.77133, 791.39154, 227.58331],
                              [631.47943, 231.76122, 645.0031, 250.34837],
                              [909.69635, 132.94533, 916.88556, 147.91328],
                              [800.7993, 171.45026, 818.24426, 190.13824],
                              [1062.6141, 94.86546, 1070.233, 102.934814],
                              [1478.5642, 344.87103, 1541.5105, 370.1643],
                              [1109.1233, 206.21417, 1127.1405, 245.65952],
                              [1185.1942, 278.27756, 1217.0431, 304.70926],
                              [1514.9675, 394.49435, 1544.4481, 428.38083],
                              [626.1507, 163.38965, 643.4621, 180.70099],
                              [950.99304, 169.18123, 960.4157, 185.39693]],
                             dtype=np.float32).tolist())

        self.assertTrue(exists)

    def test_tiny_person(self):
        data_source = DetSourceTinyPerson(
            ann_file=DET_DATASET_TINY_PERSON + '/train.json',
            img_prefix=DET_DATASET_TINY_PERSON,
            pipeline=[
                dict(type='LoadImageFromFile', to_float32=True),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=['person'],
            filter_empty_gt=False,
            iscrowd=True)

        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
