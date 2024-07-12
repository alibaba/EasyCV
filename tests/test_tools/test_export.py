# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import sys
import unittest

import numpy as np
import onnxruntime
import torch

from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import mmcv_config_fromfile, rebuild_config
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
logging.basicConfig(level=logging.INFO)

WORK_DIRECTORY = 'work_dir3'

BASIC_EXPORT_CONFIGS = {
    'config_file': None,
    'checkpoint': 'dummy',
    'output_filename': f'{WORK_DIRECTORY}/test_out.pth',
    'user_config_params': ['--export.export_type', 'onnx']
}


def build_cmd(export_configs, MODEL_TYPE) -> str:
    base_cmd = 'python tools/export.py'
    base_cmd += f" {export_configs['config_file']}"
    base_cmd += f" {export_configs['checkpoint']}"
    base_cmd += f" {export_configs['output_filename']}"
    base_cmd += f' --model_type {MODEL_TYPE}'
    user_params = ' '.join(export_configs['user_config_params'])
    base_cmd += f' --user_config_params {user_params}'
    return base_cmd


class ExportTest(unittest.TestCase):
    """In this unittest, we test the onnx export functionality of
    some classification/detection models.
    """

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        os.makedirs(WORK_DIRECTORY, exist_ok=True)

    def tearDown(self):
        super().tearDown()

    def run_test(self,
                 CONFIG_FILE,
                 MODEL_TYPE,
                 img_size: int = 224,
                 **override_configs):
        configs = BASIC_EXPORT_CONFIGS.copy()
        configs['config_file'] = CONFIG_FILE

        configs.update(override_configs)

        cmd = build_cmd(configs, MODEL_TYPE)
        logging.info(f'Export with commands: {cmd}')
        run_in_subprocess(cmd)

        cfg = mmcv_config_fromfile(configs['config_file'])
        cfg = rebuild_config(cfg, configs['user_config_params'])

        if hasattr(cfg.model, 'pretrained'):
            cfg.model.pretrained = False

        torch_model = build_model(cfg.model).eval()
        if 'checkpoint' in override_configs:
            load_checkpoint(
                torch_model,
                override_configs['checkpoint'],
                strict=False,
                logger=logging.getLogger())
        session = onnxruntime.InferenceSession(configs['output_filename'] +
                                               '.onnx')
        input_tensor = torch.randn((1, 3, img_size, img_size))

        torch_output = torch_model(input_tensor, mode='test')['prob']

        onnx_output = session.run(
            [session.get_outputs()[0].name],
            {session.get_inputs()[0].name: np.array(input_tensor)})
        if isinstance(onnx_output, list):
            onnx_output = onnx_output[0]

        onnx_output = torch.tensor(onnx_output)

        is_same_shape = torch_output.shape == onnx_output.shape

        self.assertTrue(
            is_same_shape,
            f'The shapes of the two outputs are mismatch, got {torch_output.shape} and {onnx_output.shape}'
        )
        is_allclose = torch.allclose(torch_output, onnx_output)

        torch_out_minmax = f'{float(torch_output.min())}~{float(torch_output.max())}'
        onnx_out_minmax = f'{float(onnx_output.min())}~{float(onnx_output.max())}'

        info_msg = f'got avg: {float(torch_output.mean())} and {float(onnx_output.mean())},'
        info_msg += f' and range: {torch_out_minmax} and {onnx_out_minmax}'
        self.assertTrue(
            is_allclose,
            f'The values between the two outputs are mismatch, {info_msg}')

    def test_inceptionv3(self):
        CONFIG_FILE = 'configs/classification/imagenet/inception/inceptionv3_b32x8_100e.py'
        self.run_test(CONFIG_FILE, 'CLASSIFICATION_INCEPTIONV3', 299)

    def test_inceptionv4(self):
        CONFIG_FILE = 'configs/classification/imagenet/inception/inceptionv4_b32x8_100e.py'
        self.run_test(CONFIG_FILE, 'CLASSIFICATION_INCEPTIONV4', 299)

    def test_resnext50(self):
        CONFIG_FILE = 'configs/classification/imagenet/resnext/imagenet_resnext50-32x4d_jpg.py'
        self.run_test(
            CONFIG_FILE,
            'CLASSIFICATION_RESNEXT',
            checkpoint=
            'https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/classification/resnext/resnext50-32x4d/epoch_100.pth'
        )

    def test_mobilenetv2(self):
        CONFIG_FILE = 'configs/classification/imagenet/mobilenet/mobilenetv2.py'
        self.run_test(
            CONFIG_FILE,
            'CLASSIFICATION_M0BILENET',
            checkpoint=
            'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/mobilenetv2/mobilenet_v2.pth'
        )


if __name__ == '__main__':
    unittest.main()
