# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import pickle
import shutil
import tempfile
import unittest

import cv2
import numpy as np
from mmcv import Config
from PIL import Image
from tests.ut_config import (MODEL_CONFIG_MASK2FORMER_INS,
                             MODEL_CONFIG_MASK2FORMER_PAN,
                             MODEL_CONFIG_MASK2FORMER_SEM,
                             MODEL_CONFIG_SEGFORMER,
                             PRETRAINED_MODEL_MASK2FORMER_DIR,
                             PRETRAINED_MODEL_SEGFORMER, TEST_IMAGES_DIR)

from easycv.file import io
from easycv.predictors.segmentation import (Mask2formerPredictor,
                                            SegmentationPredictor)


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

        with open(tmp_path, 'rb') as f:
            results = pickle.loads(f.read())

        self.assertEqual(len(results), total_samples)

        for res in results:
            self.assertNotIn('inputs', res)
            self.assertIn('seg_pred', res)

        shutil.rmtree(temp_dir, ignore_errors=True)


class Mask2formerPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.img_path = os.path.join(TEST_IMAGES_DIR, '000000309022.jpg')
        self.pan_ckpt = os.path.join(PRETRAINED_MODEL_MASK2FORMER_DIR,
                                     'mask2former_pan_export.pth')
        self.instance_ckpt = os.path.join(PRETRAINED_MODEL_MASK2FORMER_DIR,
                                          'mask2former_instance_export.pth')
        self.semantic_ckpt = os.path.join(PRETRAINED_MODEL_MASK2FORMER_DIR,
                                          'mask2former_semantic_export.pth')

    def test_panoptic_single(self):
        # panop
        segmentation_model_config = MODEL_CONFIG_MASK2FORMER_PAN
        predictor = Mask2formerPredictor(
            model_path=self.pan_ckpt,
            task_mode='panoptic',
            config_file=segmentation_model_config)
        img = cv2.imread(self.img_path)
        predict_out = predictor([self.img_path])
        self.assertEqual(len(predict_out), 1)
        self.assertEqual(len(predict_out[0]['masks']), 14)
        self.assertListEqual(
            predict_out[0]['labels_ids'].tolist(),
            [71, 69, 39, 39, 39, 128, 127, 122, 118, 115, 111, 104, 84, 83])

        pan_img = predictor.show_panoptic(img, **predict_out[0])
        cv2.imwrite('pan_out.jpg', pan_img)

    def test_panoptic_batch(self):
        total_samples = 2
        segmentation_model_config = MODEL_CONFIG_MASK2FORMER_PAN
        predictor = Mask2formerPredictor(
            model_path=self.pan_ckpt,
            task_mode='panoptic',
            config_file=segmentation_model_config,
            batch_size=total_samples)
        predict_out = predictor([self.img_path] * total_samples)
        self.assertEqual(len(predict_out), total_samples)
        img = cv2.imread(self.img_path)
        for i in range(total_samples):
            save_name = 'pan_out_batch_%d.jpg' % i
            self.assertEqual(len(predict_out[i]['masks']), 14)
            self.assertListEqual(predict_out[i]['labels_ids'].tolist(), [
                71, 69, 39, 39, 39, 128, 127, 122, 118, 115, 111, 104, 84, 83
            ])
            pan_img = predictor.show_panoptic(img, **predict_out[i])
            cv2.imwrite(save_name, pan_img)

    def test_instance_single(self):
        # instance
        segmentation_model_config = MODEL_CONFIG_MASK2FORMER_INS
        predictor = Mask2formerPredictor(
            model_path=self.instance_ckpt,
            task_mode='instance',
            config_file=segmentation_model_config)
        img = cv2.imread(self.img_path)
        predict_out = predictor([self.img_path])
        self.assertEqual(len(predict_out), 1)
        self.assertEqual(predict_out[0]['segms'].shape, (100, 480, 640))
        self.assertListEqual(predict_out[0]['labels'][:10].tolist(),
                             [41, 69, 72, 45, 68, 70, 41, 69, 69, 45])

        instance_img = predictor.show_instance(img, **predict_out[0])
        cv2.imwrite('instance_out.jpg', instance_img)

    def test_instance_batch(self):
        total_samples = 2
        segmentation_model_config = MODEL_CONFIG_MASK2FORMER_INS
        predictor = Mask2formerPredictor(
            model_path=self.instance_ckpt,
            task_mode='instance',
            config_file=segmentation_model_config,
            batch_size=total_samples)
        img = cv2.imread(self.img_path)
        predict_out = predictor([self.img_path] * total_samples)
        self.assertEqual(len(predict_out), total_samples)
        for i in range(total_samples):
            save_name = 'instance_out_batch_%d.jpg' % i
            self.assertEqual(predict_out[i]['segms'].shape, (100, 480, 640))
            self.assertListEqual(predict_out[0]['labels'][:10].tolist(),
                                 [41, 69, 72, 45, 68, 70, 41, 69, 69, 45])
            instance_img = predictor.show_instance(img, **(predict_out[i]))
            cv2.imwrite(save_name, instance_img)

    def test_semantic_single(self):
        # semantic
        segmentation_model_config = MODEL_CONFIG_MASK2FORMER_SEM
        predictor = Mask2formerPredictor(
            model_path=self.semantic_ckpt,
            task_mode='semantic',
            config_file=segmentation_model_config)
        img = cv2.imread(self.img_path)
        predict_out = predictor([self.img_path])
        self.assertEqual(len(predict_out), 1)
        self.assertEqual(len(np.unique(predict_out[0]['seg_pred'])), 19)

        semantic_img = predictor.show_semantic(img, **predict_out[0])
        cv2.imwrite('semantic_out.jpg', semantic_img)

    def test_semantic_batch(self):
        total_samples = 2
        segmentation_model_config = MODEL_CONFIG_MASK2FORMER_SEM
        predictor = Mask2formerPredictor(
            model_path=self.semantic_ckpt,
            task_mode='semantic',
            config_file=segmentation_model_config,
            batch_size=total_samples)
        img = cv2.imread(self.img_path)
        predict_out = predictor([self.img_path] * total_samples)
        self.assertEqual(len(predict_out), total_samples)
        for i in range(total_samples):
            save_name = 'semantic_out_batch_%d.jpg' % i
            self.assertEqual(len(np.unique(predict_out[i]['seg_pred'])), 19)
            semantic_img = predictor.show_semantic(img, **(predict_out[i]))
            cv2.imwrite(save_name, semantic_img)


if __name__ == '__main__':
    unittest.main()
