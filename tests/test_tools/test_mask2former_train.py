# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import glob
import json
import logging
import os
import sys
import tempfile
import unittest

from tests.ut_config import (DET_DATA_SMALL_COCO_LOCAL,
                             PRETRAINED_MODEL_MASK2FORMER)

from easycv.file import io
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)

SMALL_COCO_DATA_ROOT = DET_DATA_SMALL_COCO_LOCAL.rstrip('/') + '/'
_COMMON_OPTIONS = {
    'total_epochs': 1,
    'load_from': PRETRAINED_MODEL_MASK2FORMER,
    'optimizer.lr': 0.0,
    'data.imgs_per_gpu': 1,
}

TRAIN_CONFIGS = [
    {
        'config_file':
        'configs/segmentation/mask2former/mask2former_r50_8xb2_e50_instance.py',
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


class MASK2FORMERTrainTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def check_metric(self, work_dir):
        json_file = glob.glob(os.path.join(work_dir, '*.log.json'))
        with io.open(json_file[0], 'r') as f:
            content = f.readlines()
            res = json.loads(content[-1])
            # self.assertGreater(res['DetectionBoxes_Precision/mAP'], 0.4)
            # self.assertGreater(res['DetectionBoxes_Precision/mAP@.50IOU'], 0.6)
            # self.assertGreater(res['DetectionBoxes_Precision/mAP@.75IOU'], 0.5)

    def _base_train(self, train_cfgs, dist=False, dist_eval=False):
        cfg_file = train_cfgs.pop('config_file')
        cfg_options = train_cfgs.pop('cfg_options', None)
        work_dir = train_cfgs.pop('work_dir', None)
        if not work_dir:
            work_dir = tempfile.TemporaryDirectory().name

        cfg = mmcv_config_fromfile(cfg_file)
        if cfg_options is not None:
            cfg.merge_from_dict(cfg_options)

        cfg.eval_pipelines[0].data = dict(**cfg.data.val)  # imgs_per_gpu=1
        cfg.eval_pipelines[0].dist_eval = dist_eval

        # to save gpu memory avoid error
        cfg.data.train.pipeline[1].img_scale = (512, 512)
        cfg.data.train.pipeline[2].crop_size = (512, 512)
        cfg.data.train.pipeline[4].size = (512, 512)
        tmp_cfg_file = tempfile.NamedTemporaryFile(suffix='.py').name
        cfg.dump(tmp_cfg_file)

        args_str = ' '.join(
            ['='.join((str(k), str(v))) for k, v in train_cfgs.items()])

        if dist:
            nproc_per_node = 2
            cmd = 'bash tools/dist_train.sh %s %s --launcher pytorch --work_dir=%s %s' % (
                tmp_cfg_file, nproc_per_node, work_dir, args_str)
        else:
            cmd = 'python tools/train.py %s --work_dir=%s %s' % (
                tmp_cfg_file, work_dir, args_str)

        logging.info('run command: %s' % cmd)
        run_in_subprocess(cmd)

        output_files = io.listdir(work_dir)
        self.assertIn('epoch_1.pth', output_files)
        self.check_metric(work_dir)

        io.remove(work_dir)
        io.remove(tmp_cfg_file)

    def test_mask2former(self):
        train_cfgs = copy.deepcopy(TRAIN_CONFIGS[0])

        self._base_train(train_cfgs)


if __name__ == '__main__':
    unittest.main()
