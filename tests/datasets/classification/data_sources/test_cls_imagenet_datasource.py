# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

from tests.ut_config import SMALL_IMAGENET_TFRECORD_LOCAL

from easycv.datasets.builder import build_datasource


class ClsSourceImageNet1k(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_ImageNet1k(self):
        base_data_root = SMALL_IMAGENET_TFRECORD_LOCAL
        cfg = dict(type='ClsSourceImageNet1k', root=base_data_root, split='train')
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(label, list(range(10)))
            img.close()

        length = len(data_source)
        self.assertEqual(length, 50000)


if __name__ == '__main__':
    unittest.main()
