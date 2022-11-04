# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import unittest

from tests.ut_config import SAMLL_IMAGENET1K_RAW_LOCAL

from easycv.datasets.builder import build_datasource


class ClsSourceImageNet1kTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_ImageNet1k(self):
        base_data_root = SAMLL_IMAGENET1K_RAW_LOCAL

        cfg = dict(
            type='ClsSourceImageNet1k', root=base_data_root, split='train')
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertIn(label, list(range(200)))

        length = len(data_source)

        self.assertEqual(length, 1281167)


if __name__ == '__main__':
    unittest.main()
