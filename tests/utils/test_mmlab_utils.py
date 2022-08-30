# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import unittest

import torch
from mmcv.parallel import scatter_kwargs
from tests.ut_config import (COCO_CLASSES, DET_DATA_SMALL_COCO_LOCAL,
                             IMG_NORM_CFG_255)

from easycv.apis.test import single_gpu_test
from easycv.datasets import build_dataloader, build_dataset
from easycv.models.builder import build_model
from easycv.utils.config_tools import mmcv_config_fromfile
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab


class MMLabUtilTest(unittest.TestCase):

    def setUp(self):
        print(('Testing %s.%s' % (type(self).__name__, self._testMethodName)))

    def _get_model(self):
        config_path = 'configs/detection/mask_rcnn/mask_rcnn_r50_fpn.py'
        cfg = mmcv_config_fromfile(config_path)
        dynamic_adapt_for_mmlab(cfg)
        model = build_model(cfg.model)

        return model

    def _get_dataset(self, mode='train'):
        if mode == 'train':
            pipeline = [
                dict(
                    type='MMResize',
                    img_scale=[(1333, 640), (1333, 672), (1333, 704),
                               (1333, 736), (1333, 768), (1333, 800)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(type='MMRandomFlip', flip_ratio=0.5),
                dict(type='MMNormalize', **IMG_NORM_CFG_255),
                dict(type='MMPad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
                    meta_keys=('filename', 'ori_filename', 'ori_shape',
                               'ori_img_shape', 'img_shape', 'pad_shape',
                               'scale_factor', 'flip', 'flip_direction',
                               'img_norm_cfg'))
            ]
        else:
            pipeline = [
                dict(
                    type='MMMultiScaleFlipAug',
                    img_scale=(1333, 800),
                    flip=False,
                    transforms=[
                        dict(type='MMResize', keep_ratio=True),
                        dict(type='MMRandomFlip'),
                        dict(type='MMNormalize', **IMG_NORM_CFG_255),
                        dict(type='MMPad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(
                            type='Collect',
                            keys=['img'],
                            meta_keys=('filename', 'ori_filename', 'ori_shape',
                                       'ori_img_shape', 'img_shape',
                                       'pad_shape', 'scale_factor', 'flip',
                                       'flip_direction', 'img_norm_cfg')),
                    ])
            ]

        data_root = DET_DATA_SMALL_COCO_LOCAL
        dataset_cfg = dict(
            type='DetDataset',
            data_source=dict(
                type='DetSourceCoco',
                ann_file=os.path.join(data_root,
                                      'instances_train2017_20.json'),
                img_prefix=os.path.join(data_root, 'train2017'),
                pipeline=[
                    dict(type='LoadImageFromFile', to_float32=True),
                    dict(
                        type='LoadAnnotations', with_bbox=True, with_mask=True)
                ],
                classes=COCO_CLASSES,
                filter_empty_gt=False,
                iscrowd=False),
            pipeline=pipeline)
        return build_dataset(dataset_cfg)

    def test_model_train(self):
        model = self._get_model()
        model = model.cuda()
        model.train()

        dataset = self._get_dataset()
        data_loader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=1, num_gpus=1, dist=False)
        for i, data_batch in enumerate(data_loader):
            input_args, kwargs = scatter_kwargs(None, data_batch,
                                                [torch.cuda.current_device()])
            output = model(**kwargs[0], mode='train')
            self.assertEqual(len(output['loss_rpn_cls']), 5)
            self.assertEqual(len(output['loss_rpn_bbox']), 5)
            self.assertEqual(output['loss_cls'].shape, torch.Size([]))
            self.assertEqual(output['acc'].shape, torch.Size([1]))
            self.assertEqual(output['loss_bbox'].shape, torch.Size([]))
            self.assertEqual(output['loss_mask'].shape, torch.Size([1]))

    def test_model_test(self):
        model = self._get_model()
        model = model.cuda()
        dataset = self._get_dataset(mode='test')
        data_loader = build_dataloader(
            dataset, imgs_per_gpu=1, workers_per_gpu=1, num_gpus=1, dist=False)
        results = single_gpu_test(model, data_loader, mode='test')
        self.assertEqual(len(results['detection_boxes']), 20)
        self.assertEqual(len(results['detection_scores']), 20)
        self.assertEqual(len(results['detection_classes']), 20)
        self.assertEqual(len(results['detection_masks']), 20)
        self.assertEqual(len(results['img_metas']), 20)


if __name__ == '__main__':
    unittest.main()
