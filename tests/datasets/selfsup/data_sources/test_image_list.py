# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import random
import unittest

from tests.ut_config import SMALL_IMAGENET_RAW_LOCAL

from easycv.datasets.builder import build_datasource


class SSLSourceImageListTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        base_data_root = SMALL_IMAGENET_RAW_LOCAL
        cfg = dict(
            type='SSLSourceImageList',
            list_file=os.path.join(base_data_root, 'meta/train_200.txt'),
            root=base_data_root)
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            results = data_source.get_sample(idx)
            img = results['img']
            self.assertEqual(img.mode, 'RGB')
            img.close()

        self.assertEqual(len(data_source), 200)


if __name__ == '__main__':
    unittest.main()
