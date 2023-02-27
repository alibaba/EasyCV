# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import json
import os
import unittest
import numpy as np
import time
import cv2
import torch
import scipy.io
from easycv.predictors.mot_predictor import MOTPredictor
from tests.ut_config import TEST_MOT_DIR
from numpy.testing import assert_array_almost_equal


class MOTPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test(self):
        checkpoint = 'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/data/tracking/fcos_r50_epoch_12_export.pt'
        output = './result.mp4'
        imgs_folder = TEST_MOT_DIR

        model = MOTPredictor(checkpoint, score_threshold=0.2)

        model(imgs_folder, output)
