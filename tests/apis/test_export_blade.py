# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from tests.ut_config import (PRETRAINED_MODEL_RESNET50,
                             PRETRAINED_MODEL_YOLOXS_EXPORT)

from easycv.apis.export import export
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.test_util import clean_up, get_tmp_dir


@unittest.skipIf(torch.__version__ != '1.8.1+cu102',
                 'Blade need another environment')
class ModelExportTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = get_tmp_dir()
        print('tmp dir %s' % self.tmp_dir)

    def tearDown(self):
        clean_up(self.tmp_dir)

    def test_export_yolox_blade_nopre_notrt(self):
        config_file = 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg.export = dict(
            export_type='blade',
            preprocess_jit=False,
            use_trt_efficientnms=False)
        ori_ckpt = PRETRAINED_MODEL_YOLOXS_EXPORT

        target_path = f'{self.tmp_dir}/export_yolox_s_epoch300_export'

        export(cfg, ori_ckpt, target_path)

        self.assertTrue(os.path.exists(target_path + '.blade'))
        self.assertTrue(os.path.exists(target_path + '.blade.config.json'))

    def test_export_yolox_blade_pre_notrt(self):
        config_file = 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg.export = dict(
            export_type='blade',
            preprocess_jit=True,
            use_trt_efficientnms=False)

        ori_ckpt = PRETRAINED_MODEL_YOLOXS_EXPORT

        target_path = f'{self.tmp_dir}/export_yolox_s_epoch300_export'

        export(cfg, ori_ckpt, target_path)

        self.assertTrue(os.path.exists(target_path + '.blade'))
        self.assertTrue(os.path.exists(target_path + '.blade.config.json'))
        self.assertTrue(os.path.exists(target_path + '.preprocess'))


if __name__ == '__main__':
    unittest.main()
