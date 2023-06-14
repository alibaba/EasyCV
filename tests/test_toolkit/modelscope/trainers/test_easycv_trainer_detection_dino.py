# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
import tempfile
import unittest

import torch
from modelscope.msdatasets import MsDataset
from modelscope.trainers import build_trainer
from modelscope.utils.constant import LogKeys
from modelscope.utils.logger import get_logger
from modelscope.utils.test_utils import test_level


@unittest.skipIf(not torch.cuda.is_available(), 'cuda unittest')
class EasyCVTrainerTestDetectionDino(unittest.TestCase):
    model_id = 'damo/cv_swinl_image-object-detection_dino'

    def setUp(self):
        self.logger = get_logger()
        self.logger.info(
            ('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _train(self, tmp_dir):
        cfg_options = {'train.max_epochs': 1}

        trainer_name = 'easycv'

        train_dataset = MsDataset.load(
            dataset_name='small_coco_for_test',
            namespace='EasyCV',
            split='train')
        eval_dataset = MsDataset.load(
            dataset_name='small_coco_for_test',
            namespace='EasyCV',
            split='validation')

        kwargs = dict(
            model=self.model_id,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            work_dir=tmp_dir,
            use_fp16=True,
            cfg_options=cfg_options)

        trainer = build_trainer(trainer_name, kwargs)
        trainer.train()

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_trainer_single_gpu(self):
        temp_file_dir = tempfile.TemporaryDirectory()
        tmp_dir = temp_file_dir.name
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        self._train(tmp_dir)

        results_files = os.listdir(tmp_dir)
        json_files = glob.glob(os.path.join(tmp_dir, '*.log.json'))
        self.assertEqual(len(json_files), 1)
        self.assertIn(f'{LogKeys.EPOCH}_1.pth', results_files)

        temp_file_dir.cleanup()


if __name__ == '__main__':
    unittest.main()
