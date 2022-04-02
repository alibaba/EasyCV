# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

from tests.ut_config import CIFAR10_LOCAL, CIFAR100_LOCAL

from easycv.datasets.builder import build_datasource


class ClsSourceCifarTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_cifar10(self):
        base_data_root = CIFAR10_LOCAL
        cfg = dict(type='ClsSourceCifar10', root=base_data_root, split='train')
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            img, target = data_source.get_sample(idx)
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(target, list(range(10)))
            img.close()

        length = data_source.get_length()
        self.assertEqual(length, 50000)

    def test_cifar100(self):
        base_data_root = CIFAR100_LOCAL
        cfg = dict(
            type='ClsSourceCifar100', root=base_data_root, split='train')
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            img, target = data_source.get_sample(idx)
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(target, list(range(100)))
            img.close()

        length = data_source.get_length()
        self.assertEqual(length, 50000)


if __name__ == '__main__':
    unittest.main()
