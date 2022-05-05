# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from tests.ut_config import (IMAGENET_LABEL_TXT, PRETRAINED_MODEL_MOCO,
                             PRETRAINED_MODEL_RESNET50,
                             PRETRAINED_MODEL_YOLOXS)

from easycv.apis.export import export
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.test_util import clean_up, get_tmp_dir


class ModelExportTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = get_tmp_dir()
        print('tmp dir %s' % self.tmp_dir)

    def tearDown(self):
        clean_up(self.tmp_dir)

    def test_export_moco(self):
        config_file = 'configs/selfsup/mocov2/mocov2_rn50_8xb32_200e_tfrecord.py'
        ori_ckpt = PRETRAINED_MODEL_MOCO
        ckpt_path = f'{self.tmp_dir}/moco_export.pth'
        stat, output = subprocess.getstatusoutput(
            f'python tools/export.py {config_file} {ori_ckpt} {ckpt_path}')
        self.assertTrue(stat == 0, 'export model failed')
        if stat != 0:
            print(output)

    def test_export_yolox(self):
        config_file = 'configs/detection/yolox/yolox_s_8xb16_300e_voc.py'
        ori_ckpt = PRETRAINED_MODEL_YOLOXS
        ckpt_path = f'{self.tmp_dir}/export_yolox_s_epoch300.pt'
        stat, output = subprocess.getstatusoutput(
            f'python tools/export.py {config_file} {ori_ckpt} {ckpt_path}')
        self.assertTrue(stat == 0, 'export model failed')
        if stat != 0:
            print(output)

    def test_export_yolox_jit(self):
        config_file = 'configs/detection/yolox/yolox_s_8xb16_300e_voc_jit.py'
        ori_ckpt = PRETRAINED_MODEL_YOLOXS
        ckpt_path = f'{self.tmp_dir}/export_yolox_s_epoch300'
        stat, output = subprocess.getstatusoutput(
            f'python tools/export.py {config_file} {ori_ckpt} {ckpt_path}')
        self.assertTrue(stat == 0, 'export model failed')
        if stat != 0:
            print(output)

    def test_export_classification_jit(self):
        config_file = 'configs/classification/imagenet/imagenet_rn50_jpg.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg.model.backbone = dict(
            type='ResNetJIT',
            depth=50,
            out_indices=[4],
            norm_cfg=dict(type='BN'))
        cfg.export = dict(use_jit=True)
        ori_ckpt = PRETRAINED_MODEL_RESNET50
        target_ckpt = f'{self.tmp_dir}/classification.pth.jit'
        export(cfg, ori_ckpt, target_ckpt)
        self.assertTrue(os.path.exists(target_ckpt))

    def test_export_classification_and_inference(self):
        config_file = 'configs/classification/imagenet/imagenet_rn50_jpg.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg.export = dict(use_jit=False)
        ori_ckpt = PRETRAINED_MODEL_RESNET50
        target_ckpt = f'{self.tmp_dir}/classification_export.pth'
        export(cfg, ori_ckpt, target_ckpt)
        self.assertTrue(os.path.exists(target_ckpt))

        from easycv.predictors.classifier import TorchClassifier
        classifier = TorchClassifier(
            target_ckpt, label_map_path=IMAGENET_LABEL_TXT)
        img = np.random.randint(0, 255, (256, 256, 3))
        r = classifier.predict([img])

    def test_export_cls_syncbn(self):
        config_file = 'configs/classification/imagenet/imagenet_rn50_jpg.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg_options = {
            'model.backbone.norm_cfg.type': 'SyncBN',
        }
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)

        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)
        ori_ckpt = PRETRAINED_MODEL_RESNET50
        target_ckpt = f'{self.tmp_dir}/classification.pth.jit'
        export(cfg, ori_ckpt, target_ckpt)
        export_config_str = torch.load(target_ckpt)['meta']['config']
        export_config = json.loads(export_config_str)
        self.assertTrue(
            export_config['model']['backbone']['norm_cfg']['type'] == 'BN')


if __name__ == '__main__':
    unittest.main()
