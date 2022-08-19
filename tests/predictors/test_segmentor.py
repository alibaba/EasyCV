# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import os
import unittest

import numpy as np
from PIL import Image

from tests.ut_config import TEST_IMAGES_DIR
from tests.ut_config import (PRETRAINED_MODEL_SEGFORMER,
                             MODEL_CONFIG_SEGFORMER)
from easycv.predictors.segmentation import SegFormerPredictor


class SegmentorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_segformer_detector(self):
        segmentation_model_path = PRETRAINED_MODEL_SEGFORMER
        segmentation_model_config = MODEL_CONFIG_SEGFORMER

        img = os.path.join(TEST_IMAGES_DIR, '000000289059.jpg')
        if not os.path.exists(img):
            img = './data/test/segmentation/coco_stuff_164k/val2017/000000289059.jpg'

        input_data_list = [np.asarray(Image.open(img))]
        predictor = SegFormerPredictor(
            model_path=segmentation_model_path,
            model_config=segmentation_model_config)

        output = predictor.predict(input_data_list)[0]
        self.assertIn('seg_pred', output)

        self.assertListEqual(
            list(input_data_list[0].shape)[:2],
            list(output['seg_pred'][0].shape))
        self.assertListEqual(output['seg_pred'][0][1, :10].tolist(),
                             [161 for i in range(10)])
        self.assertListEqual(output['seg_pred'][0][-1, -10:].tolist(),
                             [133 for i in range(10)])


if __name__ == '__main__':
    unittest.main()
