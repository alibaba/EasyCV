# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import json
import os
import unittest

import cv2
import torch

from easycv.predictors.ocr import OCRDetPredictor, OCRRecPredictor, OCRClsPredictor, OCRPredictor

from easycv.utils.test_util import get_tmp_dir
from tests.ut_config import (PRETRAINED_MODEL_OCRDET, PRETRAINED_MODEL_OCRREC,
                             PRETRAINED_MODEL_OCRCLS, TEST_IMAGES_DIR)


class TorchOCRTest(unittest.TestCase):

    def test_ocr_det(self):
        predictor = OCRDetPredictor(PRETRAINED_MODEL_OCRDET)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_det.jpg'))
        dt_boxes = predictor([img])[0]
        self.assertEqual(dt_boxes['points'].shape[0], 16)  # 16 boxes

    def test_ocr_rec(self):
        predictor = OCRRecPredictor(PRETRAINED_MODEL_OCRREC)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_rec.jpg'))
        rec_out = predictor([img])[0]
        self.assertEqual(rec_out['preds_text'][0], '韩国小馆')  # 韩国小馆
        self.assertGreater(rec_out['preds_text'][1],
                           0.9944)  # 0.9944670796394348

    def test_ocr_direction(self):
        predictor = OCRClsPredictor(PRETRAINED_MODEL_OCRCLS)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_rec.jpg'))
        cls_out = predictor([img])[0]
        self.assertEqual(int(cls_out['class']), 0)
        self.assertGreater(float(cls_out['neck'][0]), 0.9998)  # 0.99987

    def test_ocr_end2end(self):
        predictor = OCRPredictor(
            det_model_path=PRETRAINED_MODEL_OCRDET,
            rec_model_path=PRETRAINED_MODEL_OCRREC,
            cls_model_path=PRETRAINED_MODEL_OCRCLS,
            use_angle_cls=True)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_det.jpg'))
        filter_boxes, filter_rec_res = predictor([img])
        self.assertEqual(filter_rec_res[0][0][0], '发足够的滋养')
        self.assertGreater(filter_rec_res[0][0][1], 0.91)


if __name__ == '__main__':
    unittest.main()
