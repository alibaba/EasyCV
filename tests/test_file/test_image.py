# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import unittest

import numpy as np
from tests.ut_config import TEST_IMAGES_DIR

from easycv.file.image import load_image


class LoadImageTest(unittest.TestCase):
    img_path = os.path.join(TEST_IMAGES_DIR, '000000289059.jpg')
    img_url = 'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/unittest/local_backup/easycv_nfs/data/test_images/000000289059.jpg'

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_backend_pillow(self):
        img = load_image(
            self.img_path, mode='BGR', dtype=np.float32, backend='pillow')
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(list(img[0][0]), [145, 92, 59])

    def test_backend_cv2(self):
        img = load_image(self.img_path, mode='RGB', backend='cv2')
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(list(img[0][0]), [59, 92, 145])

    def test_backend_turbojpeg(self):
        img = load_image(
            self.img_path, mode='RGB', dtype=np.float32, backend='turbojpeg')
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.float32)
        self.assertEqual(list(img[0][0]), [59, 92, 145])

    def test_url_path_cv2(self):
        img = load_image(self.img_url, mode='BGR', backend='cv2')
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(list(img[0][0]), [145, 92, 59])

    def test_url_path_pillow(self):
        img = load_image(self.img_url, mode='RGB', backend='pillow')
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(list(img[0][0]), [59, 92, 145])

    def test_url_path_turbojpeg(self):
        img = load_image(self.img_url, mode='BGR', backend='turbojpeg')
        self.assertEqual(img.shape, (480, 640, 3))
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(list(img[0][0]), [145, 92, 59])


if __name__ == '__main__':
    unittest.main()
