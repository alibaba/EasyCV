# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

from tests.ut_config import SMALL_IMAGENET_RAW_LOCAL

from easycv.datasets.builder import build_datasource
from easycv.file import io


class ClsSourceImageListByClassTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        base_data_root = SMALL_IMAGENET_RAW_LOCAL
        cfg = dict(
            type='ClsSourceImageListByClass',
            root=os.path.join(base_data_root, 'train'),
            list_file=os.path.join(base_data_root,
                                   'meta/train_labeled_200.txt'),
            m_per_class=3)
        data_source = build_datasource(cfg)

        index_list = [0, 1]
        for idx in index_list:
            img_list, target_list = data_source.get_sample(idx)
            self.assertEqual(len(img_list), 3)
            self.assertEqual(len(target_list), 3)
            self.assertEqual(img_list[0].mode, 'RGB')
            self.assertIn(target_list[0], list(range(1000)))
            for img in img_list:
                img.close()
            self.assertEqual(data_source.get_length(), 2)


if __name__ == '__main__':
    unittest.main()
