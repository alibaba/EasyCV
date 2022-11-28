# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

import numpy as np
from tests.ut_config import DET_DATASET_WIDER_FACE

from easycv.datasets.builder import build_datasource


class DetSourceWiderFaceTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source):
        index_list = random.choices(list(range(10)), k=6)
        exclude_list = [i for i in range(7) if i not in index_list]
        for idx in exclude_list:
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

        length = len(data_source)

        self.assertEqual(length, 10)

        exists = False
        for idx in range(length):

            result = data_source[idx]

            file_name = result.get('filename', '')

            if file_name.endswith('0_Parade_marchingband_1_799.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (768, 1024, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([
                        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                        2, 2, 2
                    ],
                             dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].tolist(),
                    np.array(
                        [[7.8000e+01, 2.2100e+02, 7.8700e+02, 2.2180e+03],
                         [7.8000e+01, 2.3800e+02, 7.8140e+03, 2.3817e+04],
                         [1.1300e+02, 2.1200e+02, 1.1311e+04, 2.1215e+04],
                         [1.3400e+02, 2.6000e+02, 1.3415e+04, 2.6015e+04],
                         [1.6300e+02, 2.5000e+02, 1.6314e+04, 2.5017e+04],
                         [2.0100e+02, 2.1800e+02, 2.0110e+04, 2.1812e+04],
                         [1.8200e+02, 2.6600e+02, 1.8215e+04, 2.6617e+04],
                         [2.4500e+02, 2.7900e+02, 2.4518e+04, 2.7915e+04],
                         [3.0400e+02, 2.6500e+02, 3.0416e+04, 2.6517e+04],
                         [3.2800e+02, 2.9500e+02, 3.2816e+04, 2.9520e+04],
                         [3.8900e+02, 2.8100e+02, 3.8917e+04, 2.8119e+04],
                         [4.0600e+02, 2.9300e+02, 4.0621e+04, 2.9321e+04],
                         [4.3600e+02, 2.9000e+02, 4.3622e+04, 2.9017e+04],
                         [5.2200e+02, 3.2800e+02, 5.2221e+04, 3.2818e+04],
                         [6.4300e+02, 3.2000e+02, 6.4323e+04, 3.2022e+04],
                         [6.5300e+02, 2.2400e+02, 6.5317e+04, 2.2425e+04],
                         [7.9300e+02, 3.3700e+02, 7.9323e+04, 3.3730e+04],
                         [5.3500e+02, 3.1100e+02, 5.3516e+04, 3.1117e+04],
                         [2.9000e+01, 2.2000e+02, 2.9110e+03, 2.2015e+04],
                         [3.0000e+00, 2.3200e+02, 3.1100e+02, 2.3215e+04],
                         [2.0000e+01, 2.1500e+02, 2.0120e+03, 2.1516e+04]],
                        dtype=np.float32).tolist())

        self.assertTrue(exists)

    def test_defalut(self):
        data_source = build_datasource(
            dict(
                type='DetSourceWiderFace',
                ann_file=DET_DATASET_WIDER_FACE +
                '/wider_face_train_bbx_gt.txt',
                img_prefix=DET_DATASET_WIDER_FACE + '/images',
            ))

        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
