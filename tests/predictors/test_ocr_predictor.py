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
        predictor = OCRDetPredictor(det_model_path=PRETRAINED_MODEL_OCRDET)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_det.jpg'))
        dt_boxes = predictor.predict(img)
        self.assertEqual(len(dt_boxes), 34)  # 34 boxes

    def test_ocr_rec(self):
        predictor = OCRRecPredictor(rec_model_path=PRETRAINED_MODEL_OCRREC)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_rec.jpg'))
        rec_out = predictor.predict(img)
        self.assertEqual(rec_out['preds_text'][0][0], '韩国小馆')  # 韩国小馆
        self.assertGreater(rec_out['preds_text'][0][1],
                           0.9944)  # 0.9944670796394348

    def test_ocr_direction(self):
        predictor = OCRClsPredictor(cls_model_path=PRETRAINED_MODEL_OCRCLS)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_rec.jpg'))
        _, cls_out = predictor.predict(img)
        self.assertEqual(int(cls_out['labels'][0]), 0)
        self.assertGreater(float(cls_out['logits'][0]), 0.9998)  # 0.99987

    def test_ocr_system(self):
        predictor = OCRPredictor(
            det_model_path=PRETRAINED_MODEL_OCRDET,
            rec_model_path=PRETRAINED_MODEL_OCRREC,
            cls_model_path=PRETRAINED_MODEL_OCRCLS,
            use_angle_cls=True)
        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'ocr_det.jpg'))
        filter_boxes, filter_rec_res = predictor.predict_single(img)
        self.assertEqual(len(filter_boxes), 30)
        self.assertEqual(len(filter_rec_res), 30)


if __name__ == '__main__':
    unittest.main()
