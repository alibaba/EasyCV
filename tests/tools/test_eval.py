# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import glob
import json
import logging
import os
import sys
import tempfile
import unittest

import torch
from mmcv import Config
from tests.ut_config import (DET_DATA_MANIFEST_OSS, DET_DATA_SMALL_COCO_LOCAL,
                             PRETRAINED_MODEL_YOLOXS)

from easycv.file import io
from easycv.file.utils import get_oss_config
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

SMALL_COCO_DATA_ROOT = DET_DATA_SMALL_COCO_LOCAL.rstrip('/') + '/'
SMALL_COCO_ITAG_DATA_ROOT = DET_DATA_MANIFEST_OSS.rstrip('/') + '/'
_COMMON_OPTIONS = {
    'data.imgs_per_gpu': 1,
}

TRAIN_CONFIGS = [
    # itag test
    {
        'config_file':
        'configs/detection/yolox/yolox_s_8xb16_300e_coco_pai.py',
        'cfg_options': {
            **_COMMON_OPTIONS,
            'data.train.data_source.path':
            SMALL_COCO_ITAG_DATA_ROOT + 'train2017_20.manifest',
            'data.val.data_source.path':
            SMALL_COCO_ITAG_DATA_ROOT + 'val2017_20.manifest',
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


class EvalTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def check_metric(self, work_dir):
        json_file = glob.glob(os.path.join(work_dir, '*.json'))
        with io.open(json_file[0], 'r') as f:
            content = f.readlines()
            res = json.loads(content[0])
            self.assertAlmostEqual(
                res['DetectionBoxes_Precision/mAP'], 0.423, delta=0.001)
            self.assertAlmostEqual(
                res['DetectionBoxes_Precision/mAP@.50IOU'],
                0.5816,
                delta=0.001)
            self.assertAlmostEqual(
                res['DetectionBoxes_Precision/mAP@.75IOU'], 0.451, delta=0.001)

    def _base_eval(self, eval_cfgs, dist=False, dist_eval=False):
        cfg_file = eval_cfgs.pop('config_file')
        cfg_options = eval_cfgs.pop('cfg_options', None)
        work_dir = eval_cfgs.pop('work_dir', None)
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
            ['='.join((str(k), str(v))) for k, v in eval_cfgs.items()])

        if dist:
            nproc_per_node = 2
            cmd = 'bash tools/dist_test.sh %s %s %s --eval --work_dir=%s %s ' % (
                tmp_cfg_file, nproc_per_node, PRETRAINED_MODEL_YOLOXS,
                work_dir, args_str)
        else:
            cmd = 'python tools/eval.py %s %s --eval --work_dir=%s %s' % (
                tmp_cfg_file, PRETRAINED_MODEL_YOLOXS, work_dir, args_str)

        logging.info('run command: %s' % cmd)
        run_in_subprocess(cmd)

        self.check_metric(work_dir)

        io.remove(work_dir)
        io.remove(tmp_cfg_file)

    def test_eval(self):
        eval_cfgs = copy.deepcopy(TRAIN_CONFIGS[1])

        self._base_eval(eval_cfgs)

    @unittest.skipIf(torch.cuda.device_count() <= 1, 'distributed unittest')
    def test_eval_dist(self):
        eval_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])
        eval_cfgs['cfg_options'].update(dict(oss_io_config=get_oss_config()))

        self._base_eval(eval_cfgs, dist_eval=True)


if __name__ == '__main__':
    unittest.main()
