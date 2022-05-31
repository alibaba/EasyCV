# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os
import sys
import tempfile
import unittest

from mmcv import Config
from tests.ut_config import SMALL_IMAGENET_RAW_LOCAL

from easycv.file import io
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)
SMALL_IMAGENET_DATA_ROOT = SMALL_IMAGENET_RAW_LOCAL + '/'
_COMMON_OPTIONS = {
    'checkpoint_config.interval': 1,
    'total_epochs': 1,
    'data.imgs_per_gpu': 8,
    'model.backbone.norm_cfg.type': 'BN'
}

TRAIN_CONFIGS = [{
    'config_file':
    'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py',
    'cfg_options': {
        **_COMMON_OPTIONS,
        'data.train.data_source.root':
        SMALL_IMAGENET_DATA_ROOT + 'train/',
        'data.train.data_source.list_file':
        SMALL_IMAGENET_DATA_ROOT + 'meta/train_labeled_200.txt',
        'data.val.data_source.root':
        SMALL_IMAGENET_DATA_ROOT + 'validation/',
        'data.val.data_source.list_file':
        SMALL_IMAGENET_DATA_ROOT + 'meta/val_labeled_100.txt',
    }
}, {
    'config_file':
    'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py',
    'cfg_options': {
        **_COMMON_OPTIONS, 'data.train.data_source.root':
        SMALL_IMAGENET_DATA_ROOT + 'train/',
        'data.train.data_source.list_file':
        SMALL_IMAGENET_DATA_ROOT + 'meta/train_labeled_200.txt',
        'data.val.data_source.root':
        SMALL_IMAGENET_DATA_ROOT + 'validation/',
        'data.val.data_source.list_file':
        SMALL_IMAGENET_DATA_ROOT + 'meta/val_labeled_100.txt',
        'model.train_preprocess': ['randomErasing', 'mixUp']
    }
}]


class ClassificationTrainTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def _base_train(self, train_cfgs):
        cfg_file = train_cfgs.pop('config_file')
        cfg_options = train_cfgs.pop('cfg_options', None)
        work_dir = train_cfgs.pop('work_dir', None)
        if not work_dir:
            work_dir = tempfile.TemporaryDirectory().name

        cfg = Config.fromfile(cfg_file)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)
            cfg.eval_pipelines[0].data = cfg.data.val
        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)

        args_str = ' '.join(
            ['='.join((str(k), str(v))) for k, v in train_cfgs.items()])
        cmd = 'python tools/train.py %s --work_dir=%s %s' % \
              (tmp_cfg_file, work_dir, args_str)

        logging.info('run command: %s' % cmd)
        run_in_subprocess(cmd)

        output_files = io.listdir(work_dir)
        self.assertIn('epoch_1.pth', output_files)

        io.remove(work_dir)
        io.remove(tmp_cfg_file)

    def test_classification(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])

        self._base_train(train_cfgs)

    def test_classification_mixup(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[1])

        self._base_train(train_cfgs)


if __name__ == '__main__':
    unittest.main()
