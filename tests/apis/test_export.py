# Copyright (c) Alibaba, Inc. and its affiliates.
import json
import os
import re
import subprocess
import tempfile
import unittest

import numpy as np
import torch
from tests.ut_config import (BASE_LOCAL_PATH, IMAGENET_LABEL_TXT,
                             PRETRAINED_MODEL_BEVFORMER_BASE,
                             PRETRAINED_MODEL_MOCO, PRETRAINED_MODEL_RESNET50,
                             PRETRAINED_MODEL_YOLOXS_EXPORT)

import easycv
from easycv.apis.export import export
from easycv.file import io
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
        if stat != 0:
            print(output)
        self.assertTrue(stat == 0, 'export model failed')

    def test_export_yolox(self):
        config_file = 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py'
        ori_ckpt = PRETRAINED_MODEL_YOLOXS_EXPORT
        ckpt_path = f'{self.tmp_dir}/export_yolox_s_epoch300.pt'
        stat, output = subprocess.getstatusoutput(
            f'python tools/export.py {config_file} {ori_ckpt} {ckpt_path}')
        if stat != 0:
            print(output)
        self.assertTrue(stat == 0, 'export model failed')

    def test_export_yolox_jit_nopre_notrt(self):
        config_file = 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg.export = dict(
            export_type='jit',
            preprocess_jit=False,
            use_trt_efficientnms=False)
        ori_ckpt = PRETRAINED_MODEL_YOLOXS_EXPORT

        target_path = f'{self.tmp_dir}/export_yolox_s_epoch300_export'

        export(cfg, ori_ckpt, target_path)
        self.assertTrue(os.path.exists(target_path + '.jit'))
        self.assertTrue(os.path.exists(target_path + '.jit.config.json'))

    def test_export_yolox_jit_pre_notrt(self):
        config_file = 'configs/detection/yolox/yolox_s_8xb16_300e_coco.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg.export = dict(
            export_type='jit', preprocess_jit=True, use_trt_efficientnms=False)
        ori_ckpt = PRETRAINED_MODEL_YOLOXS_EXPORT

        target_path = f'{self.tmp_dir}/export_yolox_s_epoch300_end2end'

        export(cfg, ori_ckpt, target_path)
        self.assertTrue(os.path.exists(target_path + '.jit'))
        self.assertTrue(os.path.exists(target_path + '.jit.config.json'))
        self.assertTrue(os.path.exists(target_path + '.preprocess'))

    # TOOD we will test the export of use_trt_efficientnms=True and blade in a docker environment.

    def test_export_classification_jit(self):
        config_file = 'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
        cfg = mmcv_config_fromfile(config_file)
        cfg.model.pretrained = False
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
        config_file = 'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
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
        config_file = 'configs/classification/imagenet/resnet/imagenet_resnet50_jpg.py'
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

    @unittest.skipIf(torch.__version__ != '1.8.1+cu102',
                     'need another environment where mmcv has been recompiled')
    def test_export_bevformer_jit(self):
        ckpt_path = PRETRAINED_MODEL_BEVFORMER_BASE

        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(
            config_dir,
            'detection3d/bevformer/bevformer_base_r101_dcn_nuscenes.py')

        with tempfile.TemporaryDirectory() as tmpdir:
            with io.open(config_file, 'r') as f:
                cfg_str = f.read()
            new_config_path = os.path.join(tmpdir, 'new_config.py')
            # find first adapt_jit and replace value
            res = re.search(r'adapt_jit(\s*)=(\s*)False', cfg_str)
            if res is not None:
                cfg_str_list = list(cfg_str)
                cfg_str_list[res.span()[0]:res.span()[1]] = 'adapt_jit = True'
                cfg_str = ''.join(cfg_str_list)
            with io.open(new_config_path, 'w') as f:
                f.write(cfg_str)

            cfg = mmcv_config_fromfile(new_config_path)
            cfg.export.type = 'jit'

            filename = os.path.join(tmpdir, 'model.pth')
            export(cfg, ckpt_path, filename, fp16=False)

            self.assertTrue(os.path.exists(filename + '.jit'))

    def test_export_topdown_jit(self):
        ckpt_path = os.path.join(
            BASE_LOCAL_PATH,
            'pretrained_models/pose/hrnet/pose_hrnet_epoch_210_export.pt')

        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(config_dir,
                                   'pose/hrnet_w48_coco_256x192_udp.py')

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = mmcv_config_fromfile(config_file)
            cfg.export.type = 'jit'

            filename = os.path.join(tmpdir, 'model.pth')
            export(cfg, ckpt_path, filename, fp16=False)

            self.assertTrue(os.path.exists(filename + '.jit'))

    def test_export_stgcn_jit(self):
        ckpt_path = os.path.join(
            BASE_LOCAL_PATH,
            'pretrained_models/video/stgcn/stgcn_80e_ntu60_xsub.pth')

        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(
            config_dir,
            'video_recognition/stgcn/stgcn_80e_ntu60_xsub_keypoint.py')

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = mmcv_config_fromfile(config_file)
            cfg.export.type = 'jit'

            filename = os.path.join(tmpdir, 'model.pth')
            export(cfg, ckpt_path, filename, fp16=False)

            self.assertTrue(os.path.exists(filename + '.jit'))


if __name__ == '__main__':
    unittest.main()
