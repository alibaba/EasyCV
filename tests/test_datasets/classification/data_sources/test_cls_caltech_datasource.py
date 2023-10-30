# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

from tests.ut_config import CLS_DATA_COMMON_LOCAL

from easycv.datasets.builder import build_datasource


class ClsSourceCaltechTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_caltech101(self):
        cfg = dict(
            type='ClsSourceCaltech101',
            root=CLS_DATA_COMMON_LOCAL,
            download=True)
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(label, list(range(len(data_source.CLASSES))))
            img.close()

    def test_caltech256(self):

        cfg = dict(
            type='ClsSourceCaltech256',
            root=CLS_DATA_COMMON_LOCAL,
            download=True)
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'L')
            self.assertIn(label, list(range(len(data_source.CLASSES))))
            img.close()


if __name__ == '__main__':
    unittest.main()
