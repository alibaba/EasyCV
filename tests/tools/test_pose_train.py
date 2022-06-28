# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import logging
import os
import sys
import tempfile
import unittest
import uuid

from mmcv import Config
from tests.ut_config import POSE_DATA_SMALL_COCO_LOCAL, TMP_DIR_OSS

from easycv.file import io
from easycv.file.utils import get_oss_config
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

SMALL_COCO_DATA_ROOT = POSE_DATA_SMALL_COCO_LOCAL.rstrip('/') + '/'

_COMMON_OPTIONS = {
    'checkpoint_config.interval': 1,
    'eval_config.interval': 1,
    'total_epochs': 1,
    'data.imgs_per_gpu': 16,
    'data.train.data_source.data_cfg.use_gt_bbox': True,
    'data.val.data_source.data_cfg.use_gt_bbox': True
}
TRAIN_CONFIGS = [{
    'config_file': 'configs/pose/litehrnet_30_coco_384x288.py',
    'cfg_options': {
        **_COMMON_OPTIONS, 'data.train.data_source.ann_file':
        SMALL_COCO_DATA_ROOT + 'train_200.json',
        'data.train.data_source.img_prefix':
        SMALL_COCO_DATA_ROOT + 'images/',
        'data.val.data_source.ann_file':
        SMALL_COCO_DATA_ROOT + 'val_20.json',
        'data.val.data_source.img_prefix':
        SMALL_COCO_DATA_ROOT + 'images/'
    }
}, {
    'config_file': 'configs/pose/hrnet_w48_coco_256x192_udp.py',
    'cfg_options': {
        **_COMMON_OPTIONS, 'data.train.data_source.ann_file':
        SMALL_COCO_DATA_ROOT + 'train_200.json',
        'data.train.data_source.img_prefix':
        SMALL_COCO_DATA_ROOT + 'images/',
        'data.val.data_source.ann_file':
        SMALL_COCO_DATA_ROOT + 'val_20.json',
        'data.val.data_source.img_prefix':
        SMALL_COCO_DATA_ROOT + 'images/'
    }
}]


class PoseTrainTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def _base_train(self, train_cfgs):
        io.access_oss()

        cfg_file = train_cfgs.pop('config_file')
        cfg_options = train_cfgs.pop('cfg_options', None)
        work_dir = train_cfgs.pop('work_dir', None)
        if not work_dir:
            work_dir = tempfile.TemporaryDirectory().name

        cfg = Config.fromfile(cfg_file)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)

        cfg.eval_pipelines[0].data = dict(**cfg.data.val, imgs_per_gpu=1)

        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)

        args_str = ' '.join(
            ['='.join((str(k), str(v))) for k, v in train_cfgs.items()])
        cmd = 'python tools/train.py %s --work_dir=%s %s --fp16' % \
              (tmp_cfg_file, work_dir, args_str)

        run_in_subprocess(cmd)

        output_files = io.listdir(work_dir)
        self.assertIn('epoch_1.pth', output_files)

        io.remove(work_dir)
        io.remove(tmp_cfg_file)

    # def test_litehrnet(self):
    #     train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])
    #     self._base_train(train_cfgs)

    def test_litehrnet_oss(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])

        train_cfgs['cfg_options'].update(dict(oss_io_config=get_oss_config()))
        tmp_name = uuid.uuid4().hex
        train_cfgs.update({'work_dir': os.path.join(TMP_DIR_OSS, tmp_name)})

        self._base_train(train_cfgs)

    def test_hrnet(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[1])
        self._base_train(train_cfgs)

    # def test_hrnet_oss(self):
    #     train_cfgs = copy.deepcopy(TRAIN_CONFIGS[1])

    #     train_cfgs['cfg_options'].update(
    #         dict(oss_io_config=get_oss_config()))
    #     tmp_name = uuid.uuid4().hex
    #     train_cfgs.update({'work_dir': os.path.join(
    #         TMP_DIR_OSS, tmp_name)})

    #     self._base_train(train_cfgs)


if __name__ == '__main__':
    unittest.main()
