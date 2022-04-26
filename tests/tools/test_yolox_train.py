# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os
import sys
import tempfile
import unittest

import torch
from mmcv import Config
from tests.ut_config import DET_DATA_MANIFEST_OSS, DET_DATA_SMALL_COCO_LOCAL

from easycv.file import io
from easycv.file.utils import get_oss_config
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

SMALL_COCO_DATA_ROOT = DET_DATA_SMALL_COCO_LOCAL.rstrip('/') + '/'
SMALL_COCO_ITAG_DATA_ROOT = DET_DATA_MANIFEST_OSS.rstrip('/') + '/'
_COMMON_OPTIONS = {
    'checkpoint_config.interval': 1,
    'eval_config.interval': 1,
    'total_epochs': 1,
    'data.imgs_per_gpu': 8,
}

TRAIN_CONFIGS = [
    # itag test
    {
        'config_file':
        'configs/detection/yolox/yolox_s_8xb16_300e_coco_pai.py',
        'cfg_options': {
            **_COMMON_OPTIONS, 'data.train.data_source.path':
            SMALL_COCO_ITAG_DATA_ROOT + 'train2017_20.manifest',
            'data.val.data_source.path':
            SMALL_COCO_ITAG_DATA_ROOT + 'val2017_20.manifest'
        }
    },
    {
        'config_file': 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py',
        'cfg_options': {
            **_COMMON_OPTIONS, 'data.train.data_source.img_prefix':
            SMALL_COCO_DATA_ROOT + 'train2017',
            'data.val.data_source.img_prefix':
            SMALL_COCO_DATA_ROOT + 'val2017',
            'data.train.data_source.ann_file':
            SMALL_COCO_DATA_ROOT + 'instances_train2017_20.json',
            'data.val.data_source.ann_file':
            SMALL_COCO_DATA_ROOT + 'instances_val2017_20.json'
        }
    },
]


class YOLOXTrainTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def _base_train(self, train_cfgs, dist=False, dist_eval=False):
        cfg_file = train_cfgs.pop('config_file')
        cfg_options = train_cfgs.pop('cfg_options', None)
        work_dir = train_cfgs.pop('work_dir', None)
        if not work_dir:
            work_dir = tempfile.TemporaryDirectory().name

        cfg = Config.fromfile(cfg_file)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)

        cfg.eval_pipelines[0].data = dict(**cfg.data.val)  # imgs_per_gpu=1
        cfg.eval_pipelines[0].dist_eval = dist_eval

        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)

        args_str = ' '.join(
            ['='.join((str(k), str(v))) for k, v in train_cfgs.items()])

        if dist:
            nproc_per_node = 2
            cmd = 'bash tools/dist_train.sh %s %s --launcher pytorch --work_dir=%s %s ' % (
                tmp_cfg_file, nproc_per_node, work_dir, args_str)
        else:
            cmd = 'python tools/train.py %s --work_dir=%s %s' % (
                tmp_cfg_file, work_dir, args_str)

        logging.info('run command: %s' % cmd)
        run_in_subprocess(cmd)

        output_files = io.listdir(work_dir)
        self.assertIn('epoch_1.pth', output_files)

        io.remove(work_dir)
        io.remove(tmp_cfg_file)

    def test_yolox_itag(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])
        train_cfgs['cfg_options'].update(dict(oss_io_config=get_oss_config()))

        self._base_train(train_cfgs)

    def test_yolox_coco(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[1])

        self._base_train(train_cfgs)

    @unittest.skipIf(torch.cuda.device_count() <= 1, 'distributed unittest')
    def test_yolox_itag_dist(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])
        train_cfgs['cfg_options'].update(dict(oss_io_config=get_oss_config()))

        self._base_train(train_cfgs, dist_eval=True)


if __name__ == '__main__':
    unittest.main()
