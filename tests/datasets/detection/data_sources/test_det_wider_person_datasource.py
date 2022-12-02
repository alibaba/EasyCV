# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import DET_DATASET_DOWNLOAD_WIDER_PERSON_LOCAL

from easycv.datasets.builder import build_datasource


class DetSourceWiderPerson(unittest.TestCase):

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
        self.assertEqual(length, 10)

        exists = False
        for idx in range(length):
            result = data_source[idx]
            file_name = result.get('filename', '')
            if file_name.endswith('003077.jpg'):
                exists = True
                self.assertEqual(result['img_shape'], (463, 700, 3))
                self.assertEqual(
                    result['gt_labels'].tolist(),
                    np.array([
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
                    ],
                             dtype=np.int32).tolist())
                self.assertEqual(
                    result['gt_bboxes'].astype(np.int32).tolist(),
                    np.array(
                        [[0., 176., 40., 328.], [25., 184., 84., 327.],
                         [63., 182., 124., 334.], [40., 181., 99., 325.],
                         [94., 178., 153., 324.], [122., 169., 183., 321.],
                         [159., 175., 221., 329.], [197., 177., 258., 325.],
                         [233., 172., 294., 324.], [272., 172., 336., 328.],
                         [319., 178., 380., 326.], [298., 181., 353., 318.],
                         [352., 168., 415., 322.], [401., 178., 460., 323.],
                         [381., 180., 437., 319.], [436., 184., 492., 323.],
                         [471., 175., 531., 323.], [503., 178., 563., 328.],
                         [546., 182., 601., 320.], [585., 182., 647., 334.],
                         [628., 185., 686., 327.], [96., 177., 110., 200.],
                         [165., 177., 186., 204.], [196., 173., 215., 199.],
                         [241., 178., 256., 198.], [277., 182., 295., 205.],
                         [354., 175., 376., 206.], [440., 171., 457., 197.],
                         [470., 180., 486., 202.], [509., 174., 528., 197.],
                         [548., 178., 571., 200.], [580., 178., 601., 200.],
                         [630., 178., 648., 204.]],
                        dtype=np.int32).tolist())
        self.assertTrue(exists)

    def test_default(self):

        cache_at_init = True
        cache_on_the_fly = False
        datasource_cfg = dict(
            type='DetSourceWiderPerson',
            path=os.path.join(DET_DATASET_DOWNLOAD_WIDER_PERSON_LOCAL,
                              'train.txt'),
            cache_at_init=cache_at_init,
            cache_on_the_fly=cache_on_the_fly)
        data_source = build_datasource(datasource_cfg)
        self._base_test(data_source, cache_at_init, cache_on_the_fly)


if __name__ == '__main__':
    unittest.main()
