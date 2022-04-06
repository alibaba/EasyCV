# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import json
import os
import subprocess
import unittest

import cv2
import numpy as np
import torch
from PIL import Image

from easycv.models import build_model
from easycv.predictors.feature_extractor import (TorchFaceFeatureExtractor,
                                                 TorchFeatureExtractor,
                                                 TorchMultiFaceFeatureExtractor
                                                 )
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.test_util import clean_up, get_tmp_dir
from tests.ut_config import (PRETRAINED_MODEL_RESNET50, TEST_IMAGES_DIR,
                             PRETRAINED_MODEL_MOCO, PRETRAINED_MODEL_FACEID)


class TorchMultiFaceFeatureExtractorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = get_tmp_dir()
        print('tmp dir %s' % self.tmp_dir)

    def test_torch_multi_face_feature_extractor(self):
        model_path = PRETRAINED_MODEL_FACEID
        test_img = os.path.join(TEST_IMAGES_DIR, 'multi_face.jpg')
        face_extract = TorchMultiFaceFeatureExtractor(model_path)
        img = Image.open(test_img)
        result = face_extract.predict([img])[0]
        self.assertTrue('feature' in result)
        self.assertTrue('bbox' in result)
        self.assertTrue(len(result['feature']) == 3)
        self.assertTrue(len(result['bbox']) == 3)
        self.assertTrue(result['feature'][0].reshape([1, -1]).shape[1] == 512)
        self.assertTrue(len(result['bbox'][0]) == 5)


class TorchFaceFeatureExtractorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = get_tmp_dir()
        print('tmp dir %s' % self.tmp_dir)

    def test_torch_face_feature_extractor(self):
        model_path = PRETRAINED_MODEL_FACEID
        test_img = os.path.join(TEST_IMAGES_DIR, 'multi_face.jpg')
        face_extract = TorchFaceFeatureExtractor(model_path)
        img = Image.open(test_img)
        result = face_extract.predict([img])[0]
        self.assertTrue('feature' in result)
        self.assertTrue(result['feature'].reshape([1, -1]).shape[1] == 512)


class TorchFeatureExtractorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = get_tmp_dir()
        print('tmp dir %s' % self.tmp_dir)

    def test_torch_feature_extractor(self):
        model_config = dict(
            type='Classification',
            backbone=dict(
                type='ResNet',
                depth=50,
                out_indices=[4],  # 0: conv-1, x: stage-x
                norm_cfg=dict(type='BN'),
            ),
            head=dict(
                type='ClsHead',
                with_avg_pool=True,
                in_channels=2048,
                num_classes=1000,
            ))

        img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        test_pipeline = [
            dict(type='Resize', size=[224, 224]),
            dict(type='ToTensor'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Collect', keys=['img'])
        ]

        CONFIG = dict(
            model=model_config,
            test_pipeline=test_pipeline,
        )

        meta = dict(config=json.dumps(CONFIG))
        checkpoint = PRETRAINED_MODEL_RESNET50
        state_dict = torch.load(checkpoint)['state_dict']
        output_dict = dict(state_dict=state_dict, author='EasyCV', meta=meta)
        output_ckpt = f'{self.tmp_dir}/feature_extract.pth'
        torch.save(output_dict, output_ckpt)

        fe = TorchFeatureExtractor(output_ckpt)

        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'indoor.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        feature = fe.predict([img])
        self.assertEqual(feature[0]['feature'].shape, (2048, ))
        clean_up(self.tmp_dir)

    def test_torch_feature_extractor2(self):
        config_file = 'configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py'
        ori_ckpt = PRETRAINED_MODEL_MOCO
        ckpt_path = f'{self.tmp_dir}/moco_export.pth'
        stat, output = subprocess.getstatusoutput(
            f'python tools/export.py {config_file} {ori_ckpt} {ckpt_path}')
        self.assertTrue(stat == 0, 'export model failed')
        if stat != 0:
            print(output)
        fe = TorchFeatureExtractor(ckpt_path)

        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'indoor.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        feature = fe.predict([img])
        self.assertEqual(feature[0]['feature'].shape, (2048, ))

        # check numerical equavalence
        cfg = mmcv_config_fromfile(config_file)
        model = build_model(cfg.model)
        load_checkpoint(model, ori_ckpt)
        model.eval()
        avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        ans = feature[0]['feature']
        with torch.no_grad():
            batch_image = torch.stack(fe.predictor.preprocess([img]))
            out = model(batch_image, mode='extract')['neck']
            f = avg_pool(out)
            f = torch.squeeze(f).data.cpu().numpy()
            self.assertTrue(np.allclose(ans, f, atol=1e-4))

        clean_up(self.tmp_dir)


if __name__ == '__main__':
    unittest.main()
