# Copyright (c) Alibaba, Inc. and its affiliates.
# ! -*- coding: utf8 -*-

import os
import tempfile
import unittest
import uuid

from tests.ut_config import (BASE_LOCAL_PATH, CLS_DATA_NPY_LOCAL,
                             CLS_DATA_NPY_OSS, IO_DATA_TXTX_OSS, TMP_DIR_OSS)

from easycv.file import io


class IOForOSSTest(unittest.TestCase):

    def setUp(self):
        io.access_oss()
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        pass

    def test_open(self):
        tmp_name = uuid.uuid4().hex
        test_file = os.path.join(TMP_DIR_OSS, '%s.txt' % tmp_name)
        s = 'test open'
        a = 'add'
        # test 'w' mode
        with io.open(test_file, 'w') as f:
            f.write(s)
        # test 'a' mode
        with io.open(test_file, 'a') as f:
            f.write(a)
        self.assertTrue(io.exists(test_file))
        # test 'r' mode
        with io.open(test_file, 'r') as f:
            self.assertTrue(f.read() == s + a)
        io.remove(test_file)

        # test 'wb' mode
        s = b'test open'
        with io.open(test_file, 'wb') as f:
            f.write(s)
        self.assertTrue(io.exists(test_file))
        # test 'rb' mode
        with io.open(test_file, 'rb') as f:
            self.assertTrue(f.read() == s)

        self.assertTrue(io.exists(test_file))
        io.remove(test_file)

    def test_exists(self):
        test_dir1 = CLS_DATA_NPY_LOCAL.rstrip('/') + '/'
        test_dir2 = CLS_DATA_NPY_LOCAL.rstrip('/')
        test_file = os.path.join(CLS_DATA_NPY_OSS, 'small_imagenet.npy')
        test_fake_dir1 = os.path.join(CLS_DATA_NPY_LOCAL,
                                      'fake_dir1').rstrip('/')
        test_fake_dir2 = os.path.join(CLS_DATA_NPY_LOCAL,
                                      'fake_dir2/').rstrip('/') + '/'
        test_fake_file = os.path.join(CLS_DATA_NPY_LOCAL, 'fake_file.txt')
        self.assertTrue(io.exists(test_dir1))
        self.assertTrue(io.exists(test_dir2))
        self.assertTrue(io.exists(test_file))
        self.assertFalse(io.exists(test_fake_dir1))
        self.assertFalse(io.exists(test_fake_dir2))
        self.assertFalse(io.exists(test_fake_file))

    def test_move(self):
        tmp_name = uuid.uuid4().hex
        tmp_file_name = '%s.txt' % tmp_name
        tmp_file_path = os.path.join(TMP_DIR_OSS, tmp_file_name)
        target_dir = os.path.join(TMP_DIR_OSS + 'test_move1_%s' % tmp_name)
        with io.open(tmp_file_path, 'a') as f:
            f.write('aaa')

        # test move file
        target_path = os.path.join(target_dir, tmp_file_name)
        io.move(tmp_file_path, target_path)
        self.assertFalse(io.exists(tmp_file_path))
        self.assertTrue(io.exists(target_path))
        # test move dir
        target_dir2 = os.path.join(TMP_DIR_OSS, 'test_move2_%s' % tmp_name)
        io.move(target_dir, target_dir2)
        self.assertFalse(io.exists(target_dir))
        self.assertTrue(io.exists(os.path.join(target_dir2, tmp_file_name)))
        io.remove(target_dir2)

    def test_copy(self):
        # test copy file from oss to local
        oss_file_path1 = os.path.join(IO_DATA_TXTX_OSS, 'a.txt')
        temp_dir = tempfile.TemporaryDirectory().name
        tmp_path = os.path.join(temp_dir, 'a.txt')
        io.copy(oss_file_path1, tmp_path)
        self.assertTrue(io.exists(tmp_path))

        # test copy file from local to oss
        oss_file_path2 = os.path.join(TMP_DIR_OSS,
                                      '%s/a.txt' % uuid.uuid4().hex)
        io.copy(tmp_path, oss_file_path2)
        self.assertTrue(io.exists(oss_file_path2))

        io.remove(temp_dir)
        io.remove(oss_file_path2)

        # test copy file from local to oss
        oss_file_path3 = os.path.join(TMP_DIR_OSS,
                                      '%s/a.txt' % uuid.uuid4().hex)
        io.copy(oss_file_path1, oss_file_path3)
        self.assertTrue(io.exists(oss_file_path3))

        io.remove(oss_file_path3)

    def test_copytree(self):
        # test copy dir from oss to local
        oss_file_path1 = IO_DATA_TXTX_OSS
        temp_dir = tempfile.TemporaryDirectory().name
        io.copytree(oss_file_path1, temp_dir)
        self.assertTrue(io.exists(temp_dir))
        self.assertCountEqual(io.listdir(temp_dir), ['a.txt', 'b.txt'])

        # test copy dir from local to oss
        oss_file_path2 = os.path.join(TMP_DIR_OSS, '%s' % uuid.uuid4().hex)
        io.copytree(temp_dir, oss_file_path2)
        self.assertTrue(io.exists(oss_file_path2))
        self.assertCountEqual(io.listdir(oss_file_path2), ['a.txt', 'b.txt'])

        io.remove(temp_dir)
        io.remove(oss_file_path2)

    def test_listdir(self):
        # with suffix /
        files = io.listdir(IO_DATA_TXTX_OSS.rstrip('/') + '/')
        self.assertCountEqual(files, ['a.txt', 'b.txt'])
        # without suffix /
        files = io.listdir(IO_DATA_TXTX_OSS.rstrip('/'))
        self.assertCountEqual(files, ['a.txt', 'b.txt'])

    def test_isdir(self):
        self.assertTrue(io.isdir(IO_DATA_TXTX_OSS.rstrip('/') + '/'))
        self.assertTrue(io.isdir(IO_DATA_TXTX_OSS.rstrip('/')))
        self.assertFalse(io.isdir(os.path.join(IO_DATA_TXTX_OSS, 'a.txt')))

    def test_isfile(self):
        self.assertFalse(io.isfile(IO_DATA_TXTX_OSS.rstrip('/') + '/'))
        self.assertFalse(io.isfile(IO_DATA_TXTX_OSS.rstrip('/')))
        self.assertTrue(io.isfile(os.path.join(IO_DATA_TXTX_OSS, 'a.txt')))

    def test_glob(self):
        files_list = io.glob(os.path.join(IO_DATA_TXTX_OSS, '*.txt'))
        self.assertCountEqual(
            files_list,
            [os.path.join(IO_DATA_TXTX_OSS, i) for i in ['a.txt', 'b.txt']])

    def test_exist_middle_dirs(self):
        tmp_dir = os.path.join(TMP_DIR_OSS, uuid.uuid4().hex)
        io.copytree(IO_DATA_TXTX_OSS, os.path.join(tmp_dir, 'tmp1/tmp2/tmp3'))

        self.assertTrue(io.exists(os.path.join(tmp_dir, 'tmp1')))
        self.assertTrue(io.exists(os.path.join(tmp_dir, 'tmp1/tmp2/')))
        self.assertTrue(io.exists(os.path.join(tmp_dir, 'tmp1/tmp2/tmp3')))
        self.assertTrue(
            io.exists(os.path.join(tmp_dir, 'tmp1/tmp2/tmp3/a.txt')))

        io.remove(tmp_dir)


class IOForLocalTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        pass

    def test_open(self):
        tmp_file = tempfile.NamedTemporaryFile(suffix='.txt').name
        s = 'test open'
        a = 'add'
        # test 'w' mode
        with io.open(tmp_file, 'w') as f:
            f.write(s)
        # test 'a' mode
        with io.open(tmp_file, 'a') as f:
            f.write(a)
        self.assertTrue(io.exists(tmp_file))
        # test 'r' mode
        with io.open(tmp_file, 'r') as f:
            self.assertTrue(f.read() == s + a)
        io.remove(tmp_file)

        # test 'wb' mode
        s = b'test open'
        with io.open(tmp_file, 'wb') as f:
            f.write(s)
        self.assertTrue(io.exists(tmp_file))
        # test 'rb' mode
        with io.open(tmp_file, 'rb') as f:
            self.assertTrue(f.read() == s)

        self.assertTrue(io.exists(tmp_file))
        io.remove(tmp_file)

    def test_exists(self):
        test_dir = CLS_DATA_NPY_LOCAL
        test_file = os.path.join(CLS_DATA_NPY_LOCAL, 'small_imagenet.npy')
        test_fake_dir = BASE_LOCAL_PATH + 'fake_dir1'
        test_fake_file = BASE_LOCAL_PATH + 'fake_file.txt'
        self.assertTrue(io.exists(test_dir))
        self.assertTrue(io.exists(test_file))
        self.assertFalse(io.exists(test_fake_dir))
        self.assertFalse(io.exists(test_fake_file))

    def test_move(self):
        tmp_dir = tempfile.TemporaryDirectory().name
        tmp_file_path = os.path.join(tmp_dir, 'a.txt')
        io.makedirs(tmp_dir)
        with io.open(tmp_file_path, 'a') as f:
            f.write('aaa')

        # test move file
        tmp_dir2 = tempfile.TemporaryDirectory().name
        io.makedirs(tmp_dir2)
        target_path = os.path.join(tmp_dir2, 'a.txt')
        io.move(tmp_file_path, target_path)
        self.assertFalse(io.exists(tmp_file_path))
        self.assertTrue(io.exists(target_path))
        # test move dir

        io.move(tmp_dir, tmp_dir2)
        self.assertFalse(io.exists(tmp_dir))
        self.assertTrue(io.exists(os.path.join(tmp_dir2, 'a.txt')))
        io.remove(tmp_dir2)

    def test_copy(self):
        # test copy file from oss to local
        file_path = os.path.join(CLS_DATA_NPY_LOCAL, 'small_imagenet.npy')
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, 'a.npy')
            io.copy(file_path, tmp_path)
            self.assertTrue(io.exists(tmp_path))

    def test_copytree(self):
        file_dir = CLS_DATA_NPY_LOCAL
        with tempfile.TemporaryDirectory() as tmp_dir:
            io.copytree(file_dir, tmp_dir)
            self.assertTrue(io.exists(tmp_dir))
            self.assertCountEqual(
                io.listdir(tmp_dir),
                ['small_imagenet_label.npy', 'small_imagenet.npy'])

    def test_isdir(self):
        file_dir = CLS_DATA_NPY_LOCAL
        self.assertTrue(io.isdir(file_dir))
        self.assertFalse(
            io.isdir(os.path.join(file_dir, 'small_imagenet.npy')))

    def test_isfile(self):
        file_dir = CLS_DATA_NPY_LOCAL
        self.assertFalse(io.isfile(file_dir))
        self.assertTrue(
            io.isfile(os.path.join(file_dir, 'small_imagenet.npy')))


if __name__ == '__main__':
    unittest.main()
