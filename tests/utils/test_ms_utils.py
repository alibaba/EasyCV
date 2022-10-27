# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import shutil
import tempfile
import unittest

import easycv
from easycv.utils.config_tools import Config
from easycv.utils.ms_utils import to_ms_config


class MsConfigTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def tearDown(self):
        super().tearDown()
        shutil.rmtree(self.tmp_dir)

    def test_to_ms_config(self):

        config_path = os.path.join(
            os.path.dirname(easycv.__file__),
            'configs/detection/yolox/yolox_s_8xb16_300e_coco.py')

        ms_cfg_file = os.path.join(self.tmp_dir,
                                   'ms_yolox_s_8xb16_300e_coco.json')
        to_ms_config(
            config_path,
            task='image-object-detection',
            ms_model_name='yolox',
            pipeline_name='easycv-detection',
            reserved_keys=['CLASSES'],
            save_path=ms_cfg_file)
        cfg = Config.fromfile(ms_cfg_file)
        self.assertIn('task', cfg)
        self.assertIn('framework', cfg)
        self.assertIn('CLASSES', cfg)
        self.assertIn('preprocessor', cfg)
        self.assertIn('pipeline', cfg)
        self.assertEqual(cfg.model.type, 'yolox')
        self.assertIn('dataset', cfg)
        self.assertIn('batch_size_per_gpu', cfg.train.dataloader)
        self.assertIn('batch_size_per_gpu', cfg.evaluation.dataloader)


if __name__ == '__main__':
    unittest.main()
