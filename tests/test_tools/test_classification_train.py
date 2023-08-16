# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os
import sys
import tempfile
import unittest

from tests.ut_config import CLS_TRAIN_TEST

from easycv.file import io
from easycv.utils.config_tools import mmcv_config_fromfile, pai_config_fromfile
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)
_COMMON_OPTIONS = {
    'checkpoint_config.interval': 1,
    'total_epochs': 1,
    'data.imgs_per_gpu': 8,
    'model.backbone.norm_cfg.type': 'BN',
    'model.head.num_classes': 2,
}

TRAIN_CONFIGS = [{
    'config_file':
    'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py',
    'cfg_options': {
        **_COMMON_OPTIONS,
        'data.train.data_source.root': '',
        'data.train.data_source.list_file': CLS_TRAIN_TEST,
        'data.val.data_source.root': '',
        'data.val.data_source.list_file': CLS_TRAIN_TEST,
        'data.train.data_source.class_list': ['ok', 'ng'],
        'data.val.data_source.class_list': ['ok', 'ng'],
    }
}, {
    'config_file':
    'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py',
    'cfg_options': {
        **_COMMON_OPTIONS,
        'data.train.data_source.root': '',
        'data.train.data_source.list_file': CLS_TRAIN_TEST,
        'data.val.data_source.root': '',
        'data.val.data_source.list_file': CLS_TRAIN_TEST,
        'model.train_preprocess': ['randomErasing', 'mixUp'],
        'data.train.data_source.class_list': ['ok', 'ng'],
        'data.val.data_source.class_list': ['ok', 'ng'],
    }
}, {
    'config_file':
    'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py',
    'cfg_options': {
        **_COMMON_OPTIONS,
        'data_train_root': '',
        'data_train_list': CLS_TRAIN_TEST,
        'data_test_root': '',
        'data_test_list': CLS_TRAIN_TEST,
        'image_resize2': [224, 224],
        'save_epochs': 1,
        'eval_epochs': 1,
        'class_list': ['ok', 'ng'],
    }
}]


class ClassificationTrainTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def _base_train(self, train_cfgs, adapt_pai=False):
        cfg_file = train_cfgs.pop('config_file')
        cfg_options = train_cfgs.pop('cfg_options', None)
        work_dir = train_cfgs.pop('work_dir', None)
        if not work_dir:
            work_dir = tempfile.TemporaryDirectory().name

        if adapt_pai:
            cfg = pai_config_fromfile(cfg_file, user_config_params=cfg_options)
            cfg.eval_pipelines[0].data = cfg.data.val
        else:
            cfg = mmcv_config_fromfile(cfg_file)
            if cfg_options is not None:
                cfg.merge_from_dict(cfg_options)
                cfg.eval_pipelines[0].data = cfg.data.val

        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)

        args_str = ' '.join(
            ['='.join((str(k), str(v))) for k, v in train_cfgs.items()])
        cmd = 'python tools/train.py %s --work_dir=%s %s --fp16' % \
              (tmp_cfg_file, work_dir, args_str)

        logging.info('run command: %s' % cmd)
        # run_in_subprocess(cmd)  # 管道缓冲区被写满，后面的写入请求都hang住了
        import subprocess
        subprocess.call(cmd, shell=True)

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

    def test_classification_pai(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[2])

        self._base_train(train_cfgs, adapt_pai=True)


if __name__ == '__main__':
    unittest.main()
