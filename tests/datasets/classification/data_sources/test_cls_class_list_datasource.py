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
            results = data_source[idx]
            img_list = results['img']
            label_list = results['gt_labels']
            self.assertEqual(len(img_list), 3)
            self.assertEqual(len(label_list), 3)
            self.assertEqual(img_list[0].mode, 'RGB')
            self.assertIn(label_list[0], list(range(1000)))
            for img in img_list:
                img.close()
            self.assertEqual(len(data_source), 2)


if __name__ == '__main__':
    unittest.main()
