import os
import unittest

import numpy as np
from PIL import Image
from tests.ut_config import TEST_IMAGES_DIR

from easycv.predictors.builder import build_predictor
from easycv.thirdparty.u2sod.sodpredictor import SODPredictor

bbox_res = [[0, 1077, 1, 1079], [147, 871, 148, 873], [172, 196, 197, 218],
            [266, 68, 267, 70], [676, 0, 679, 2], [104, 0, 507, 1008]]


class SODPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_sodpredictor(self):
        sodpredictor = SODPredictor(model_name='u2netp')
        image = Image.open(os.path.join(TEST_IMAGES_DIR, 'u2sod_bottle.jpg'))
        # bboxes, landmarks = detector.detect(image)
        res = sodpredictor.predict([image])
        bbox = np.array(res[0]['bbox'])
        # self.assertTrue(np.allclose(bbox, np.array(bbox_res)))

    def test_ev_sodpredictor(self):
        sod_cfg = dict(type='SODPredictor', model_name='u2netp')
        sodpredictor = build_predictor(sod_cfg)
        image = Image.open(os.path.join(TEST_IMAGES_DIR, 'u2sod_bottle.jpg'))
        # bboxes, landmarks = detector.detect(image)
        res = sodpredictor.predict([image])
        bbox = np.array(res[0]['bbox'])
        # self.assertTrue(np.allclose(bbox, np.array(bbox_res)))


if __name__ == '__main__':
    unittest.main()
