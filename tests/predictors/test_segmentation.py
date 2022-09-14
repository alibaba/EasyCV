# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from PIL import Image
from tests.ut_config import (MODEL_CONFIG_SEGFORMER,
                             PRETRAINED_MODEL_SEGFORMER, TEST_IMAGES_DIR)

from easycv.predictors.segmentation import SegmentationPredictor


class SegmentationPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def test_single(self):
        segmentation_model_path = PRETRAINED_MODEL_SEGFORMER
        segmentation_model_config = MODEL_CONFIG_SEGFORMER

        img_path = os.path.join(TEST_IMAGES_DIR, '000000289059.jpg')
        img = np.asarray(Image.open(img_path))

        predict_pipeline = SegmentationPredictor(
            model_path=segmentation_model_path,
            config_file=segmentation_model_config)

        outputs = predict_pipeline(img_path, keep_inputs=True)
        self.assertEqual(len(outputs), 1)
        self.assertEqual(outputs[0]['inputs'], [img_path])

        results = outputs[0]['results']
        self.assertListEqual(
            list(img.shape)[:2], list(results['seg_pred'][0].shape))
        self.assertListEqual(results['seg_pred'][0][1, :10].tolist(),
                             [161 for i in range(10)])
        self.assertListEqual(results['seg_pred'][0][-1, -10:].tolist(),
                             [133 for i in range(10)])

    def test_batch(self):
        segmentation_model_path = PRETRAINED_MODEL_SEGFORMER
        segmentation_model_config = MODEL_CONFIG_SEGFORMER

        img_path = os.path.join(TEST_IMAGES_DIR, '000000289059.jpg')
        img = np.asarray(Image.open(img_path))

        predict_pipeline = SegmentationPredictor(
            model_path=segmentation_model_path,
            config_file=segmentation_model_config,
            batch_size=2)

        total_samples = 3
        outputs = predict_pipeline(
            [img_path] * total_samples, keep_inputs=True)
        self.assertEqual(len(outputs), 2)

        self.assertEqual(outputs[0]['inputs'], [img_path] * 2)
        self.assertEqual(outputs[1]['inputs'], [img_path] * 1)
        self.assertEqual(len(outputs[0]['results']['seg_pred']), 2)
        self.assertEqual(len(outputs[1]['results']['seg_pred']), 1)

        for result in [outputs[0]['results'], outputs[1]['results']]:
            self.assertListEqual(
                list(img.shape)[:2], list(result['seg_pred'][0].shape))
            self.assertListEqual(result['seg_pred'][0][1, :10].tolist(),
                                 [161 for i in range(10)])
            self.assertListEqual(result['seg_pred'][0][-1, -10:].tolist(),
                                 [133 for i in range(10)])

    def test_dump(self):
        segmentation_model_path = PRETRAINED_MODEL_SEGFORMER
        segmentation_model_config = MODEL_CONFIG_SEGFORMER

        img_path = os.path.join(TEST_IMAGES_DIR, '000000289059.jpg')

        temp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        tmp_path = os.path.join(temp_dir, 'results.pkl')

        predict_pipeline = SegmentationPredictor(
            model_path=segmentation_model_path,
            config_file=segmentation_model_config,
            batch_size=2,
            save_results=True,
            save_path=tmp_path)

        total_samples = 3
        outputs = predict_pipeline(
            [img_path] * total_samples, keep_inputs=True)
        self.assertEqual(outputs, [])

        with open(tmp_path, 'rb') as f:
            results = pickle.loads(f.read())

        self.assertIn('inputs', results[0])
        self.assertIn('results', results[0])

        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
