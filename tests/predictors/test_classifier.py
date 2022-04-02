# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import json
import os
import unittest

import cv2
import torch

from easycv.predictors.classifier import TorchClassifier

from easycv.utils.test_util import clean_up, get_tmp_dir
from tests.ut_config import (PRETRAINED_MODEL_RESNET50_WITHOUTHEAD,
                             IMAGENET_LABEL_TXT, TEST_IMAGES_DIR)


class TorchClassifierTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = get_tmp_dir()
        print('tmp dir %s' % self.tmp_dir)

    def test_torch_classifier(self, topk=5):
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
        ]

        CONFIG = dict(
            model=model_config,
            test_pipeline=test_pipeline,
            export_neck=True,
        )

        meta = dict(config=json.dumps(CONFIG))
        checkpoint = PRETRAINED_MODEL_RESNET50_WITHOUTHEAD
        state_dict = torch.load(checkpoint)['state_dict']
        output_dict = dict(state_dict=state_dict, author='EasyCV', meta=meta)

        output_ckpt = f'{self.tmp_dir}/export.pth'
        torch.save(output_dict, output_ckpt)

        fe = TorchClassifier(
            output_ckpt, topk=topk, label_map_path=IMAGENET_LABEL_TXT)

        img = cv2.imread(os.path.join(TEST_IMAGES_DIR, 'catb.jpg'))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        feature = fe.predict([img])
        self.assertEqual(feature[0]['class'][0],
                         283)  # imagenet 283 = tiger cat
        clean_up(self.tmp_dir)

    def test_torch_classifier_top1(self):
        self.test_torch_classifier(topk=1)

    def test_classifier_topk_overflow(self):
        self.test_torch_classifier(topk=1001)


if __name__ == '__main__':
    unittest.main()
