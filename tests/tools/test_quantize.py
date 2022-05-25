# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os
import sys
import tempfile
import unittest

import torch
from mmcv import Config
from tests.ut_config import (COMPRESSION_TEST_DATA,
                             PRETRAINED_MODEL_YOLOX_COMPRESSION)

from easycv.file import io
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

SMALL_IMAGENET_DATA_ROOT = COMPRESSION_TEST_DATA.rstrip('/') + '/'
_QUANTIZE_OPTIONS = {
    'total_epochs': 1,
    'data.imgs_per_gpu': 16,
}

_PRUNE_OPTIONS = {
    'total_epochs': 11,
    'data.imgs_per_gpu': 1,
}

TRAIN_CONFIGS = [
    {
        'config_file': 'configs/edge_models/yolox_edge.py',
        'model_type': 'YOLOX_EDGE',
        'cfg_options': {
            **_QUANTIZE_OPTIONS, 'data.train.data_source.ann_file':
            SMALL_IMAGENET_DATA_ROOT + 'annotations/instances_train2017.json',
            'data.train.data_source.img_prefix':
            SMALL_IMAGENET_DATA_ROOT + 'images',
            'data.val.data_source.ann_file':
            SMALL_IMAGENET_DATA_ROOT + 'annotations/instances_train2017.json',
            'data.val.data_source.img_prefix':
            SMALL_IMAGENET_DATA_ROOT + 'images'
        }
    },
    {
        'config_file': 'configs/edge_models/yolox_edge.py',
        'model_type': 'YOLOX_EDGE',
        'cfg_options': {
            **_PRUNE_OPTIONS, 'img_scale': (128, 128),
            'data.train.data_source.ann_file':
            SMALL_IMAGENET_DATA_ROOT + 'annotations/instances_train2017.json',
            'data.train.data_source.img_prefix':
            SMALL_IMAGENET_DATA_ROOT + 'images',
            'data.val.data_source.ann_file':
            SMALL_IMAGENET_DATA_ROOT + 'annotations/instances_train2017.json',
            'data.val.data_source.img_prefix':
            SMALL_IMAGENET_DATA_ROOT + 'images'
        }
    },
]


class ModelQuantizeTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def _base_quantize(self, train_cfgs):
        cfg_file = train_cfgs.pop('config_file')
        cfg_options = train_cfgs.pop('cfg_options', None)
        work_dir = train_cfgs.pop('work_dir', None)
        model_type = train_cfgs.pop('model_type', None)
        if not work_dir:
            work_dir = tempfile.TemporaryDirectory().name

        cfg = Config.fromfile(cfg_file)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)
        cfg.eval_pipelines[0].data = dict(**cfg.data.val)

        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)

        ckpt_path = PRETRAINED_MODEL_YOLOX_COMPRESSION

        args_str = ' '.join(
            ['='.join((str(k), str(v))) for k, v in train_cfgs.items()])

        cmd = 'python tools/quantize.py %s %s --model_type=%s --work_dir=%s %s' % \
              (tmp_cfg_file, ckpt_path, model_type, work_dir, args_str)

        logging.info('run command: %s' % cmd)
        run_in_subprocess(cmd)

        output_files = io.listdir(work_dir)
        self.assertIn('quantize_model.pt', output_files)

        io.remove(work_dir)
        io.remove(tmp_cfg_file)

    @unittest.skipIf(torch.__version__ < '1.8.0',
                     'model compression need pytorch version >= 1.8.0')
    def test_model_quantize(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])

        self._base_quantize(train_cfgs)


if __name__ == '__main__':
    unittest.main()
