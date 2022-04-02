# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import tempfile
import unittest

from tests.ut_config import CLS_DATA_NPY_LOCAL, CLS_DATA_NPY_OSS

from easycv.datasets.builder import build_datasource
from easycv.file import io


class ImageNpyTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_default(self):
        base_data_root = CLS_DATA_NPY_LOCAL
        cfg = dict(
            type='ImageNpy',
            image_file=base_data_root + 'small_imagenet.npy',
            label_file=base_data_root + 'small_imagenet_label.npy')
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            img, target = data_source.get_sample(idx)
            self.assertEqual(img.mode, 'RGB')
            img.close()
            self.assertIn(target, list(range(1000)))
            self.assertEqual(data_source.get_length(), 100)

    def test_oss(self):
        io.access_oss()

        base_data_root = CLS_DATA_NPY_OSS
        work_dir = tempfile.TemporaryDirectory().name
        cfg = dict(
            type='ImageNpy',
            image_file=base_data_root + 'small_imagenet.npy',
            label_file=base_data_root + 'small_imagenet_label.npy',
            cache_root=work_dir)
        data_source = build_datasource(cfg)

        index_list = random.choices(list(range(100)), k=3)
        for idx in index_list:
            img, target = data_source.get_sample(idx)
            self.assertEqual(img.mode, 'RGB')
            img.close()
            self.assertIn(target, list(range(1000)))
            self.assertEqual(data_source.get_length(), 100)

        io.remove(work_dir)


if __name__ == '__main__':
    unittest.main()
