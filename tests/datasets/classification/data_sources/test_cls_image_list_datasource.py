# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

from tests.ut_config import SMALL_IMAGENET_RAW_LOCAL

from easycv.datasets.builder import build_datasource


class ClsSourceImageListTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        base_data_root = SMALL_IMAGENET_RAW_LOCAL
        cfg = dict(
            type='ClsSourceImageList',
            root=os.path.join(base_data_root, 'train'),
            list_file=os.path.join(base_data_root,
                                   'meta/train_labeled_200.txt'))
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(label, list(range(1000)))
            img.close()

        self.assertEqual(len(data_source), 200)


if __name__ == '__main__':
    unittest.main()
