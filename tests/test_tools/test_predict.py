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
from tests.ut_config import (PRETRAINED_MODEL_SEGFORMER,
                             PRETRAINED_MODEL_YOLOXS_EXPORT, TEST_IMAGES_DIR)

from easycv.file import get_oss_config, io
from easycv.utils.test_util import run_in_subprocess

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

logging.basicConfig(level=logging.INFO)


class PredictTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self):
        super().tearDown()

    def _base_predict(self, model_type, model_path, dist=False):
        input_file = tempfile.NamedTemporaryFile('w').name
        input_line_num = 10
        with open(input_file, 'w') as ofile:
            for _ in range(input_line_num):
                ofile.write(
                    os.path.join(TEST_IMAGES_DIR, '000000289059.jpg') + '\n')
        output_file = tempfile.NamedTemporaryFile('w').name

        if dist:
            cmd = f'PYTHONPATH=. python -m torch.distributed.launch --nproc_per_node=2 --master_port=29527 \
                    tools/predict.py \
                    --input_file {input_file} \
                    --output_file {output_file} \
                    --model_type {model_type} \
                    --model_path {model_path} \
                    --launcher pytorch'

        else:
            cmd = f'PYTHONPATH=. python tools/predict.py \
                    --input_file {input_file} \
                    --output_file {output_file} \
                    --model_type {model_type} \
                    --model_path {model_path} '

        logging.info('run command: %s' % cmd)
        run_in_subprocess(cmd)

        with open(output_file, 'r') as infile:
            output_line_num = len(infile.readlines())
        self.assertEqual(input_line_num, output_line_num)

        io.remove(input_file)
        io.remove(output_file)

    def test_predict(self):
        model_type = 'YoloXPredictor'
        model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        self._base_predict(model_type, model_path)

    @unittest.skipIf(torch.cuda.device_count() <= 1, 'distributed unittest')
    def test_predict_dist(self):
        model_type = 'YoloXPredictor'
        model_path = PRETRAINED_MODEL_YOLOXS_EXPORT
        self._base_predict(model_type, model_path, dist=True)

    def test_predict_oss_path(self):
        model_type = 'YoloXPredictor'
        model_path = PRETRAINED_MODEL_YOLOXS_EXPORT

        os.environ['OSS_CONFIG_FILE'] = '~/.ossutilconfig.unittest'
        oss_config = get_oss_config()
        ak_id = oss_config['ak_id']
        ak_secret = oss_config['ak_secret']
        hosts = oss_config['hosts']
        hosts = ','.join(_ for _ in hosts)
        buckets = oss_config['buckets']
        buckets.append('easycv')
        buckets = ','.join(_ for _ in buckets)

        input_file = 'oss://easycv/data/small_imagenet_raw/meta/val_labeled_100_ap_no_label.txt'
        output_file = tempfile.NamedTemporaryFile('w').name
        cmd = f'PYTHONPATH=. python tools/predict.py \
                    --input_file {input_file} \
                    --output_file {output_file} \
                    --model_type {model_type} \
                    --model_path {model_path} \
                    --oss_io_config ak_id={ak_id} ak_secret={ak_secret} hosts={hosts} buckets={buckets}'

        logging.info('run command: %s' % cmd)
        run_in_subprocess(cmd)


if __name__ == '__main__':
    unittest.main()
