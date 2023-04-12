# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import cv2
import numpy as np
from PIL import Image
from tests.ut_config import TEST_IMAGES_DIR

from easycv.datasets.shared.pipelines.transforms import LoadImage


class LoadImageTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _check_results(self, results):
        shape = (1350, 2408, 3)
        self.assertEqual(results['img_shape'], shape)
        self.assertEqual(results['ori_shape'], shape)
        self.assertListEqual(results['img_fields'], ['img'])
        self.assertEqual(results['img'].shape, shape)

    def test_load_np(self):
        load_op = LoadImage()
        img_path = os.path.join(TEST_IMAGES_DIR, 'multi_face.jpg')
        inputs = {'img': cv2.imread(img_path)}
        results = load_op(inputs)
        self._check_results(results)
        self.assertEqual(results['filename'], None)
        self.assertEqual(results['img'].dtype, np.uint8)

    def test_load_pil(self):
        load_op = LoadImage(to_float32=True)
        img_path = os.path.join(TEST_IMAGES_DIR, 'multi_face.jpg')
        inputs = {'img': Image.open(img_path)}
        results = load_op(inputs)
        self._check_results(results)
        self.assertEqual(results['filename'], None)
        self.assertEqual(results['img'].dtype, np.float32)

    def test_load_path(self):
        load_op = LoadImage(to_float32=True)
        img_path = os.path.join(TEST_IMAGES_DIR, 'multi_face.jpg')
        inputs = {'filename': img_path}
        results = load_op(inputs)
        self._check_results(results)
        self.assertEqual(results['filename'], img_path)
        self.assertEqual(results['img'].dtype, np.float32)


if __name__ == '__main__':
    unittest.main()
