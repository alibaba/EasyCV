# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from mmcv.parallel import scatter_kwargs
from tests.ut_config import SMALL_NUSCENES_PATH

import easycv
from easycv.datasets import build_dataloader, build_dataset
from easycv.models import build_model
from easycv.models.detection3d.detectors import BEVFormer
from easycv.utils.config_tools import mmcv_config_fromfile


class BEVFormerTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _prepare_data_and_model(self, test=False, tiny=False):
        easycv_dir = os.path.dirname(easycv.__file__)
        if os.path.exists(os.path.join(easycv_dir, 'configs')):
            config_dir = os.path.join(easycv_dir, 'configs')
        else:
            config_dir = os.path.join(os.path.dirname(easycv_dir), 'configs')
        if tiny:
            config_file = os.path.join(
                config_dir,
                'detection3d/bevformer/bevformer_tiny_r50_nuscenes.py')
        else:
            config_file = os.path.join(
                config_dir,
                'detection3d/bevformer/bevformer_base_r101_dcn_nuscenes.py')
        cfg = mmcv_config_fromfile(config_file)
        if test:
            cfg.data.val.data_source.data_root = SMALL_NUSCENES_PATH
            cfg.data.val.data_source.ann_file = os.path.join(
                SMALL_NUSCENES_PATH, 'nuscenes_infos_temporal_train_20.pkl')
            cfg.data.val.pop('imgs_per_gpu', None)
            dataset = build_dataset(cfg.data.val)
            shuffle = False
        else:
            cfg.data.train.data_source.data_root = SMALL_NUSCENES_PATH
            cfg.data.train.data_source.ann_file = os.path.join(
                SMALL_NUSCENES_PATH, 'nuscenes_infos_temporal_train_20.pkl')
            dataset = build_dataset(cfg.data.train)
            shuffle = True
        dataloader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=1, shuffle=shuffle)
        model = build_model(cfg.model)
        return dataloader, model

    def test_bevformer_tiny_train(self):
        dataloader, model = self._prepare_data_and_model(tiny=True)
        model = model.cuda()
        model.train()

        self.assertIsInstance(model, BEVFormer)

        for i, data in enumerate(dataloader):
            data, _ = scatter_kwargs(data, None, [torch.cuda.current_device()])
            output = model.train_step(data[0], optimizer=None)
            self.assertIn('loss', output)
            self.assertIn('loss_cls', output['log_vars'])
            self.assertIn('loss_bbox', output['log_vars'])
            self.assertIn('d0.loss_bbox', output['log_vars'])
            self.assertIn('d0.loss_cls', output['log_vars'])
            self.assertIn('d1.loss_bbox', output['log_vars'])
            self.assertIn('d1.loss_cls', output['log_vars'])
            self.assertIn('d2.loss_bbox', output['log_vars'])
            self.assertIn('d2.loss_cls', output['log_vars'])
            self.assertIn('d3.loss_bbox', output['log_vars'])
            self.assertIn('d3.loss_cls', output['log_vars'])
            self.assertIn('d4.loss_bbox', output['log_vars'])
            self.assertIn('d4.loss_cls', output['log_vars'])

            break

    def test_bevformer_base_test(self):
        dataloader, model = self._prepare_data_and_model(test=True)
        model = model.cuda()
        model.eval()

        self.assertIsInstance(model, BEVFormer)

        for i, data in enumerate(dataloader):
            data, _ = scatter_kwargs(data, None, [torch.cuda.current_device()])
            with torch.no_grad():
                outputs = model(**data[0], mode='test')
            self.assertEqual(len(outputs), 1)
            output = outputs[0]['pts_bbox']
            self.assertEqual(output['boxes_3d'].bev.shape, torch.Size([300,
                                                                       5]))
            self.assertEqual(output['boxes_3d'].bottom_center.shape,
                             torch.Size([300, 3]))
            self.assertEqual(output['boxes_3d'].bottom_height.shape,
                             torch.Size([300]))
            self.assertEqual(output['boxes_3d'].center.shape,
                             torch.Size([300, 3]))
            self.assertEqual(output['boxes_3d'].corners.shape,
                             torch.Size([300, 8, 3]))
            self.assertEqual(output['boxes_3d'].dims.shape,
                             torch.Size([300, 3]))
            self.assertEqual(output['boxes_3d'].gravity_center.shape,
                             torch.Size([300, 3]))
            self.assertEqual(output['boxes_3d'].height.shape,
                             torch.Size([300]))
            self.assertEqual(output['boxes_3d'].nearest_bev.shape,
                             torch.Size([300, 4]))
            self.assertEqual(output['boxes_3d'].volume.shape,
                             torch.Size([300]))
            self.assertEqual(output['boxes_3d'].yaw.shape, torch.Size([300]))
            self.assertEqual(output['boxes_3d'].top_height.shape,
                             torch.Size([300]))
            self.assertEqual(output['scores_3d'].shape, torch.Size([300]))
            self.assertEqual(output['labels_3d'].shape, torch.Size([300]))

            break


if __name__ == '__main__':
    unittest.main()
