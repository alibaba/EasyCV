import os
import unittest
import uuid

from tests.ut_config import SMALL_IMAGENET_TFRECORD_OSS, TMP_DIR_LOCAL

from easycv.datasets.utils import download_tfrecord
from easycv.file import io


class DistDownloadTfrecordTest(unittest.TestCase):

    def setUp(self):
        io.access_oss()
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_download_tfrecord_0(self):
        file_path = os.path.join(SMALL_IMAGENET_TFRECORD_OSS, 'meta/train.txt')

        target_path0 = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        path0, index_path0 = download_tfrecord(
            file_list_or_path=file_path,
            target_path=target_path0,
            slice_count=2,
            slice_id=0)

        with io.open(file_path, 'r') as f:
            oss_sample_list = [i.strip() for i in f.readlines()]
        name_list = [os.path.split(i)[-1] for i in oss_sample_list]
        target_path_list = [os.path.join(target_path0, i) for i in name_list]
        target_index_list = [
            os.path.join(target_path0, i) + '.idx' for i in name_list
        ]

        self.assertCountEqual(path0, target_path_list)
        self.assertCountEqual(index_path0, target_index_list)
        self.assertEqual(len(io.listdir(target_path0)), 4)

        io.remove(target_path0)

        target_path1 = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        path1, index_path1 = download_tfrecord(
            file_list_or_path=file_path,
            target_path=target_path1,
            slice_count=2,
            slice_id=1)

        name_list = [os.path.split(i)[-1] for i in oss_sample_list]
        target_path_list = [os.path.join(target_path1, i) for i in name_list]
        target_index_list = [
            os.path.join(target_path1, i) + '.idx' for i in name_list
        ]

        self.assertCountEqual(path1, target_path_list)
        self.assertCountEqual(index_path1, target_index_list)
        self.assertEqual(len(io.listdir(target_path1)), 2)

        io.remove(target_path1)

    def test_download_tfrecord_1(self):
        file_path = os.path.join(SMALL_IMAGENET_TFRECORD_OSS, 'meta/train.txt')

        target_path0 = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        path0, index_path0 = download_tfrecord(
            file_list_or_path=file_path,
            target_path=target_path0,
            slice_count=2,
            slice_id=0)
        with io.open(file_path, 'r') as f:
            oss_sample_list = [i.strip() for i in f.readlines()]
        name_list = [os.path.split(i)[-1] for i in oss_sample_list]
        target_index_list = [
            os.path.join(target_path0, i) + '.idx' for i in name_list
        ]
        target_path_list = [os.path.join(target_path0, i) for i in name_list]

        self.assertCountEqual(path0, target_path_list)
        self.assertCountEqual(index_path0, target_index_list)
        self.assertEqual(len(io.listdir(target_path0)), 4)

        io.remove(target_path0)

        target_path1 = os.path.join(TMP_DIR_LOCAL, uuid.uuid4().hex)
        path1, index_path1 = download_tfrecord(
            file_list_or_path=file_path,
            target_path=target_path1,
            slice_count=2,
            slice_id=1)

        name_list = [os.path.split(i)[-1] for i in oss_sample_list]
        target_index_list = [
            os.path.join(target_path1, i) + '.idx' for i in name_list
        ]
        target_path_list = [os.path.join(target_path1, i) for i in name_list]
        self.assertCountEqual(path1, target_path_list)
        self.assertCountEqual(index_path1, target_index_list)
        self.assertEqual(len(io.listdir(target_path1)), 2)

        io.remove(target_path1)


if __name__ == '__main__':
    unittest.main()
