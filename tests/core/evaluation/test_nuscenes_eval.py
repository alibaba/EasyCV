# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from mmcv.parallel import scatter_kwargs
from tests.ut_config import NUSCENES_CLASSES, SMALL_NUSCENES_PATH

import easycv
from easycv.core.evaluation import NuScenesEvaluator
from easycv.datasets import build_dataloader, build_dataset
from easycv.models import build_model
from easycv.utils.config_tools import mmcv_config_fromfile


class NuScenesEvaluatorTest(unittest.TestCase):
    tmp_dir = None

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def tearDown(self) -> None:
        if self.tmp_dir is not None:
            self.tmp_dir.cleanup()
        return super().tearDown()

    def _prepare_data_and_model(self):
        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(
            config_dir, 'detection3d/bevformer/bevformer_tiny_r50_nuscenes.py')
        cfg = mmcv_config_fromfile(config_file)

        cfg.data.val.data_source.data_root = SMALL_NUSCENES_PATH
        cfg.data.val.data_source.ann_file = os.path.join(
            SMALL_NUSCENES_PATH, 'nuscenes_infos_temporal_val.pkl')
        cfg.data.val.pop('imgs_per_gpu', None)
        dataset = build_dataset(cfg.data.val)
        model = build_model(cfg.model)
        return dataset, model

    def test_evaluator(self):
        from nuscenes import NuScenes

        dataset, model = self._prepare_data_and_model()
        model = model.cuda()
        model.eval()
        dataloader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=1, shuffle=False)
        evaluator = NuScenesEvaluator(classes=NUSCENES_CLASSES)

        outputs_list = []
        for i, data in enumerate(dataloader):
            data, _ = scatter_kwargs(data, None, [torch.cuda.current_device()])
            with torch.no_grad():
                outputs = model(**data[0], mode='test')
            outputs_list.extend(outputs)

        result_files, self.tmp_dir = dataset.format_results(outputs_list)
        nusc = NuScenes(
            version=dataset.data_source.version,
            dataroot=dataset.data_source.data_root,
            verbose=True)

        res = evaluator.evaluate(
            result_files,
            nusc,
            eval_detection_configs=dataset.eval_detection_configs)

        self.assertIn('pts_bbox_NuScenes/NDS', res)
        self.assertIn('pts_bbox_NuScenes/mAP', res)


if __name__ == '__main__':
    unittest.main()
