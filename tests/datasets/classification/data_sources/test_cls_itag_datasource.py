# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

from tests.ut_config import CLS_DATA_ITAG_OSS

from easycv.datasets.builder import build_datasource
from easycv.framework.errors import ValueError


class ClsSourceImageListTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        from easycv.file import io
        io.access_oss()

        cfg = dict(type='ClsSourceItag', list_file=CLS_DATA_ITAG_OSS)
        data_source = build_datasource(cfg)

        index_list = list(range(5))
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(label, list(range(3)))
            img.close()

        self.assertEqual(len(data_source), 11)
        self.assertDictEqual(data_source.label_dict, {
            'ng': 0,
            'ok': 1,
            '中文': 2
        })

    def test_with_class_list(self):
        from easycv.file import io
        io.access_oss()

        cfg = dict(
            type='ClsSourceItag',
            class_list=['中文', 'ng', 'ok'],
            list_file=CLS_DATA_ITAG_OSS)
        data_source = build_datasource(cfg)

        index_list = list(range(5))
        for idx in index_list:
            results = data_source[idx]
            img = results['img']
            label = results['gt_labels']
            self.assertEqual(img.mode, 'RGB')
            self.assertIn(label, list(range(3)))
            img.close()

        self.assertEqual(len(data_source), 11)
        self.assertDictEqual(data_source.label_dict, {
            '中文': 0,
            'ng': 1,
            'ok': 2
        })

    def test_with_fault_class_list(self):
        from easycv.file import io
        io.access_oss()

        with self.assertRaises(ValueError) as cm:
            cfg = dict(
                type='ClsSourceItag',
                class_list=['error', 'ng', 'ok'],
                list_file=CLS_DATA_ITAG_OSS)

            data_source = build_datasource(cfg)
            index_list = list(range(5))
            for idx in index_list:
                results = data_source[idx]
                img = results['img']
                label = results['gt_labels']
                self.assertEqual(img.mode, 'RGB')
                self.assertIn(label, list(range(3)))
                img.close()

        exception = cm.exception
        self.assertEqual(
            exception.message,
            "Not find label \"中文\" in label dict: {'error': 0, 'ng': 1, 'ok': 2}"
        )


if __name__ == '__main__':
    unittest.main()
