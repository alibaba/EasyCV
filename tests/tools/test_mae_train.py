# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os
import sys
import tempfile
import unittest
from distutils.version import LooseVersion

import torch
from tests.ut_config import (PRETRAINED_MODEL_MAE, SMALL_IMAGENET_RAW_LOCAL,
                             SSL_SMALL_IMAGENET_RAW)

from easycv.file import io
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

SSL_SMALL_IMAGENET_DATA_ROOT = SSL_SMALL_IMAGENET_RAW.rstrip('/') + '/'
SMALL_IMAGENET_DATA_ROOT = SMALL_IMAGENET_RAW_LOCAL.rstrip('/') + '/'
_COMMON_OPTIONS = {
    'checkpoint_config.interval': 1,
    'total_epochs': 1,
    'data.imgs_per_gpu': 8,
}

TRAIN_CONFIGS = [
    {
        'config_file':
        'configs/selfsup/mae/mae_vit_base_patch16_8xb64_400e.py',
        'cfg_options': {
            **_COMMON_OPTIONS, 'data.train.data_source.root':
            SSL_SMALL_IMAGENET_DATA_ROOT,
            'data.train.data_source.list_file':
            SSL_SMALL_IMAGENET_DATA_ROOT + 'train_20.txt'
        }
    },
    {
        'config_file':
        'benchmarks/selfsup/classification/imagenet/mae_vit_base_patch16_8xb64_100e_lrdecay065_fintune.py',
        'cfg_options': {
            **_COMMON_OPTIONS, 'data.train.data_source.root':
            SMALL_IMAGENET_DATA_ROOT + 'train/',
            'data.train.data_source.list_file':
            SMALL_IMAGENET_DATA_ROOT + 'meta/train_labeled_200.txt',
            'data.val.data_source.root':
            SMALL_IMAGENET_DATA_ROOT + 'validation/',
            'data.val.data_source.list_file':
            SMALL_IMAGENET_DATA_ROOT + 'meta/val_labeled_100.txt',
            'model.pretrained':
            PRETRAINED_MODEL_MAE
        }
    },
]


@unittest.skipIf(
    LooseVersion(torch.__version__) < LooseVersion('1.6.0'),
    'adapt fp16, not support torch.cuda.amp below 1.6.0 ')
class MAETrainTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def _base_train(self, train_cfgs):
        cfg_file = train_cfgs.pop('config_file')
        cfg_options = train_cfgs.pop('cfg_options', None)
        print(cfg_options)
        work_dir = train_cfgs.pop('work_dir', None)
        if not work_dir:
            work_dir = tempfile.TemporaryDirectory().name

        cfg = mmcv_config_fromfile(cfg_file)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)
            if 'eval_pipelines' in cfg:
                cfg.eval_pipelines[0].data = cfg.data.val

        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)

        args_str = ' '.join(
            ['='.join((str(k), str(v))) for k, v in train_cfgs.items()])
        cmd = 'python tools/train.py %s --fp16 --work_dir=%s %s' % \
              (tmp_cfg_file, work_dir, args_str)

        logging.info('run command: %s' % cmd)

        run_in_subprocess(cmd)

        output_files = io.listdir(work_dir)
        self.assertIn('epoch_1.pth', output_files)

        io.remove(work_dir)
        io.remove(tmp_cfg_file)

    def test_mae_pretrain(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])

        self._base_train(train_cfgs)

    def test_mae_fintune(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[1])

        self._base_train(train_cfgs)


if __name__ == '__main__':
    unittest.main()
