# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

import numpy as np
from tests.ut_config import (COCO_CLASSES, COCO_DATASET_DOWNLOAD_SMALL,
                             DET_DATA_SMALL_COCO_LOCAL)

from easycv.datasets.segmentation.data_sources.coco import (SegSourceCoco,
                                                            SegSourceCoco2017)


class SegSourceCocoTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _base_test(self, data_source):
        index_list = random.choices(list(range(20)), k=3)
        for idx in index_list:
            data = data_source[idx]
            self.assertIn('filename', data)
            self.assertEqual(data['img_fields'], ['img'])
            self.assertEqual(data['seg_fields'], ['gt_semantic_seg'])
            self.assertIn('img_shape', data)
            self.assertEqual(len(data['img_shape']), 3)
            self.assertEqual(data['gt_semantic_seg'].shape,
                             data['img_shape'][:2])
            self.assertEqual(data['img'].shape[-1], 3)
            self.assertTrue(
                set([255]).issubset(np.unique(data['gt_semantic_seg'])))
            self.assertTrue(
                len(np.unique(data['gt_semantic_seg'])) < len(COCO_CLASSES))

        length = len(data_source)
        self.assertEqual(length, 20)
        self.assertEqual(data_source.PALETTE.shape, (len(COCO_CLASSES), 3))

    def test_seg_source_coco(self):
        data_root = DET_DATA_SMALL_COCO_LOCAL

        data_source = SegSourceCoco(
            ann_file=os.path.join(data_root, 'instances_train2017_20.json'),
            img_prefix=os.path.join(data_root, 'train2017'),
            reduce_zero_label=True)

        self._base_test(data_source)

    def test_seg_download_coco(self):

        data_source = SegSourceCoco2017(
            download=True,
            split='train',
            path=COCO_DATASET_DOWNLOAD_SMALL,
            reduce_zero_label=True)

        self._base_test(data_source)


if __name__ == '__main__':
    unittest.main()
