# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import re
import tempfile
import unittest

import mmcv
import numpy as np
import torch
from numpy.testing import assert_array_almost_equal
from tests.ut_config import (PRETRAINED_MODEL_BEVFORMER_BASE,
                             SMALL_NUSCENES_PATH)

import easycv
from easycv.apis.export import export
from easycv.core.evaluation.builder import build_evaluator
from easycv.datasets import build_dataset
from easycv.file import io
from easycv.predictors import BEVFormerPredictor
from easycv.utils.config_tools import mmcv_config_fromfile


class BEVFormerPredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _assert_results(self, results):
        res = results['pts_bbox']
        self.assertEqual(res['scores_3d'].shape, torch.Size([300]))
        self.assertEqual(res['labels_3d'].shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].bev.shape, torch.Size([300, 5]))
        self.assertEqual(res['boxes_3d'].bottom_center.shape,
                         torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].bottom_height.shape,
                         torch.Size([300]))
        self.assertEqual(res['boxes_3d'].center.shape, torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].corners.shape, torch.Size([300, 8,
                                                                    3]))
        self.assertEqual(res['boxes_3d'].dims.shape, torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].gravity_center.shape,
                         torch.Size([300, 3]))
        self.assertEqual(res['boxes_3d'].height.shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].nearest_bev.shape,
                         torch.Size([300, 4]))
        self.assertEqual(res['boxes_3d'].tensor.shape, torch.Size([300, 9]))
        self.assertEqual(res['boxes_3d'].top_height.shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].volume.shape, torch.Size([300]))
        self.assertEqual(res['boxes_3d'].yaw.shape, torch.Size([300]))

    def _get_config_file(self):
        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(
            config_dir,
            'detection3d/bevformer/bevformer_base_r101_dcn_nuscenes.py')
        return config_file

    def test_single(self):
        model_path = PRETRAINED_MODEL_BEVFORMER_BASE
        single_ann_file = os.path.join(SMALL_NUSCENES_PATH,
                                       'inference/single_sample.pkl')
        config_file = self._get_config_file()
        predictor = BEVFormerPredictor(
            model_path=model_path,
            config_file=config_file,
        )
        results = predictor(single_ann_file)
        self.assertEqual(len(results), 1)
        for result in results:
            self._assert_results(result)

    @unittest.skipIf(True, 'Not support batch yet')
    def test_batch(self):
        model_path = PRETRAINED_MODEL_BEVFORMER_BASE
        single_ann_file = os.path.join(SMALL_NUSCENES_PATH,
                                       'inference/single_sample.pkl')
        config_file = self._get_config_file()
        predictor = BEVFormerPredictor(
            model_path=model_path, config_file=config_file, batch_size=2)
        results = predictor([single_ann_file, single_ann_file])
        self.assertEqual(len(results), 2)
        # Input the same sample continuously, the output value is different,
        # because the model will record the features of the previous sample to infer the next sample
        self._assert_results(results[0])
        self._assert_results(results[1])

    def test_metric(self):
        model_path = PRETRAINED_MODEL_BEVFORMER_BASE
        inputs_file = os.path.join(SMALL_NUSCENES_PATH,
                                   'nuscenes_infos_temporal_val.pkl')
        config_file = self._get_config_file()
        cfg = mmcv_config_fromfile(config_file)
        cfg.data.val.data_source.data_root = SMALL_NUSCENES_PATH
        cfg.data.val.data_source.ann_file = os.path.join(
            SMALL_NUSCENES_PATH, 'nuscenes_infos_temporal_val.pkl')
        cfg.data.val.pop('imgs_per_gpu', None)
        val_dataset = build_dataset(cfg.data.val)
        evaluators = build_evaluator(cfg.eval_pipelines[0]['evaluators'][0])
        predictor = BEVFormerPredictor(
            model_path=model_path, config_file=config_file)
        inputs = mmcv.load(inputs_file)['infos']
        for i in range(len(inputs)):
            for k in list(inputs[i]['cams'].keys()):
                inputs[i]['cams'][k]['data_path'] = os.path.join(
                    SMALL_NUSCENES_PATH, inputs[i]['cams'][k]['data_path'])
        predict_results = predictor(inputs)

        results = {'pts_bbox': [i['pts_bbox'] for i in predict_results]}
        val_results = val_dataset.evaluate(results, evaluators)
        self.assertAlmostEqual(
            val_results['pts_bbox_NuScenes/NDS'], 0.460, delta=0.01)
        self.assertAlmostEqual(
            val_results['pts_bbox_NuScenes/mAP'], 0.41, delta=0.01)


@unittest.skipIf(torch.__version__ != '1.8.1+cu102',
                 'need another environment where mmcv has been recompiled')
class BEVFormerBladePredictorTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))
        self.tmp_dir = tempfile.TemporaryDirectory().name
        if not os.path.exists(self.tmp_dir):
            os.mkdir(self.tmp_dir)

    def tearDown(self) -> None:
        io.remove(self.tmp_dir)
        return super().tearDown()

    def _replace_config(self, cfg_file):
        with io.open(cfg_file, 'r') as f:
            cfg_str = f.read()

        new_config_path = os.path.join(self.tmp_dir, 'new_config.py')

        # find first adapt_jit and replace value
        res = re.search(r'adapt_jit(\s*)=(\s*)False', cfg_str)
        if res is not None:
            cfg_str_list = list(cfg_str)
            cfg_str_list[res.span()[0]:res.span()[1]] = 'adapt_jit = True'
            cfg_str = ''.join(cfg_str_list)
        with io.open(new_config_path, 'w') as f:
            f.write(cfg_str)
        return new_config_path

    def test_single(self):
        # test export blade model and bevformer predictor
        ori_ckpt = PRETRAINED_MODEL_BEVFORMER_BASE
        inputs_file = os.path.join(SMALL_NUSCENES_PATH,
                                   'nuscenes_infos_temporal_val.pkl')

        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        config_file = os.path.join(
            config_dir,
            'detection3d/bevformer/bevformer_base_r101_dcn_nuscenes.py')
        config_file = self._replace_config(config_file)
        cfg = mmcv_config_fromfile(config_file)

        filename = os.path.join(self.tmp_dir, 'model.pth')
        export(cfg, ori_ckpt, filename, fp16=False)
        blade_filename = filename + '.blade'

        self.assertTrue(blade_filename)

        cfg.data.val.data_source.data_root = SMALL_NUSCENES_PATH
        cfg.data.val.data_source.ann_file = os.path.join(
            SMALL_NUSCENES_PATH, 'nuscenes_infos_temporal_val.pkl')
        cfg.data.val.pop('imgs_per_gpu', None)
        val_dataset = build_dataset(cfg.data.val)
        evaluators = build_evaluator(cfg.eval_pipelines[0]['evaluators'][0])

        predictor = BEVFormerPredictor(
            model_path=blade_filename,
            config_file=config_file,
            model_type='blade',
        )

        inputs = mmcv.load(inputs_file)['infos']
        predict_results = predictor(inputs)

        results = {'pts_bbox': [i['pts_bbox'] for i in predict_results]}
        val_results = val_dataset.evaluate(results, evaluators)
        self.assertAlmostEqual(
            val_results['pts_bbox_NuScenes/NDS'], 0.460, delta=0.01)
        self.assertAlmostEqual(
            val_results['pts_bbox_NuScenes/mAP'], 0.41, delta=0.01)

    @unittest.skipIf(True, 'Not support batch yet')
    def test_batch(self):
        pass


if __name__ == '__main__':
    unittest.main()
