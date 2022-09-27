# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle
import shutil
import tempfile
import unittest

import numpy as np
from mmcv import Config
from PIL import Image
from tests.ut_config import (MODEL_CONFIG_SEGFORMER,
                             PRETRAINED_MODEL_MASK2FORMER_DIR,
                             PRETRAINED_MODEL_SEGFORMER, TEST_IMAGES_DIR)

from easycv.file import io
from easycv.predictors.segmentation import SegmentationPredictor


class SegmentationPredictorTest(unittest.TestCase):
    img_path = os.path.join(TEST_IMAGES_DIR, '000000289059.jpg')

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _assert_results(self, results, img_path):
        self.assertEqual(results['inputs'], img_path)
        img = np.asarray(Image.open(img_path))
        self.assertListEqual(
            list(img.shape)[:2], list(results['seg_pred'].shape))
        self.assertListEqual(results['seg_pred'][1, :10].tolist(),
                             [161 for i in range(10)])
        self.assertListEqual(results['seg_pred'][-1, -10:].tolist(),
                             [133 for i in range(10)])

    def test_single_cpu(self):
        cfg_file = MODEL_CONFIG_SEGFORMER
        model_path = PRETRAINED_MODEL_SEGFORMER

        with tempfile.NamedTemporaryFile(suffix='.py') as tmppath:
            with io.open(cfg_file, 'r') as f:
                cfg_str = f.read()

            cfg_str = cfg_str.replace('SyncBN', 'BN')
            with io.open(tmppath.name, 'w') as f:
                f.write(cfg_str)

            cfg = Config.fromfile(tmppath.name)

        predict_pipeline = SegmentationPredictor(
            model_path=model_path, config_file=cfg, device='cpu')

        outputs = predict_pipeline(self.img_path, keep_inputs=True)
        self.assertEqual(len(outputs), 1)
        results = outputs[0]
        self._assert_results(results, self.img_path)

    def test_single(self):
        segmentation_model_path = PRETRAINED_MODEL_SEGFORMER
        segmentation_model_config = MODEL_CONFIG_SEGFORMER
        predict_pipeline = SegmentationPredictor(
            model_path=segmentation_model_path,
            config_file=segmentation_model_config)

        outputs = predict_pipeline(self.img_path, keep_inputs=True)
        self.assertEqual(len(outputs), 1)
        results = outputs[0]
        self._assert_results(results, self.img_path)

    def test_batch(self):
        segmentation_model_path = PRETRAINED_MODEL_SEGFORMER
        segmentation_model_config = MODEL_CONFIG_SEGFORMER
        predict_pipeline = SegmentationPredictor(
            model_path=segmentation_model_path,
            config_file=segmentation_model_config,
            batch_size=2)

        total_samples = 3
        outputs = predict_pipeline(
            [self.img_path] * total_samples, keep_inputs=True)
        self.assertEqual(len(outputs), 3)

        for i in range(len(outputs)):
            self._assert_results(outputs[i], self.img_path)

    def test_dump(self):
        segmentation_model_path = PRETRAINED_MODEL_SEGFORMER
        segmentation_model_config = MODEL_CONFIG_SEGFORMER
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
            [self.img_path] * total_samples, keep_inputs=False)
        self.assertEqual(outputs, [])

        with open(tmp_path, 'rb') as f:
            results = pickle.loads(f.read())

        for res in results:
            self.assertNotIn('inputs', res)
            self.assertIn('seg_pred', res)

        shutil.rmtree(temp_dir, ignore_errors=True)


@unittest.skipIf(True, 'WIP')
class Mask2formerPredictorTest(unittest.TestCase):

    def test_single(self):
        import cv2
        from easycv.predictors.segmentation import Mask2formerPredictor
        pan_ckpt = os.path.join(PRETRAINED_MODEL_MASK2FORMER_DIR,
                                'mask2former_pan_export.pth')
        instance_ckpt = os.path.join(PRETRAINED_MODEL_MASK2FORMER_DIR,
                                     'mask2former_r50_instance.pth')
        img_path = os.path.join(TEST_IMAGES_DIR, 'mask2former.jpg')

        # panop
        predictor = Mask2formerPredictor(
            model_path=pan_ckpt, output_mode='panoptic')
        img = cv2.imread(img_path)
        predict_out = predictor([img])
        pan_img = predictor.show_panoptic(img, predict_out[0]['pan'])
        cv2.imwrite('pan_out.jpg', pan_img)

        # instance
        predictor = Mask2formerPredictor(
            model_path=instance_ckpt, output_mode='instance')
        img = cv2.imread(img_path)
        predict_out = predictor.predict([img], mode='instance')
        instance_img = predictor.show_instance(img, **predict_out[0])
        cv2.imwrite('instance_out.jpg', instance_img)


if __name__ == '__main__':
    unittest.main()
