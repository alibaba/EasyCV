# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

from tests.ut_config import CLS_DATA_COMMON_LOCAL

from easycv.datasets.builder import build_datasource


class ClsSourceMnistTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_mnist(self):
        cfg = dict(
            type='ClsSourceMnist',
            root=CLS_DATA_COMMON_LOCAL,
            split='train',
            download=True)
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(label, list(range(10)))
            img.close()

    def test_fashionmnist(self):

        cfg = dict(
            type='ClsSourceFashionMnist',
            root=CLS_DATA_COMMON_LOCAL,
            split='train',
            download=True)
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(label, list(range(100)))
            img.close()


if __name__ == '__main__':
    unittest.main()
