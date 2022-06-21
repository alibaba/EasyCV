# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.registry import MODELS
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger, print_log


@MODELS.register_module
class FCOS(BaseModel):

    def __init__(self, backbone, head=None, neck=None, pretrained=None):
        super(FCOS, self).__init__()

        self.fp16_enabled = False
        self.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)

        self.init_weights()

    @property
    def with_neck(self):
        """bool: whether the detector has a neck"""
        return hasattr(self, 'neck') and self.neck is not None

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self.backbone, self.pretrained, strict=False, logger=logger)
        else:
            print_log('load model from init weights')
            if self.backbone.__class__.__name__ == 'PytorchImageModelWrapper':
                self.backbone.init_weights(pretrained=self.pretrained)
            else:
                self.backbone.init_weights()
        self.neck.init_weights()
        self.head.init_weights()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        # print(x)
        x = self.neck(x)
        # print(x)
        return x

    def forward_train(self, imgs, img_metas, gt_bboxes, gt_labels):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        batch_input_shape = tuple(imgs[0].size()[-2:])
        for i in range(len(img_metas)):
            img_metas[i]['batch_input_shape'] = batch_input_shape

        x = self.extract_feat(imgs)
        losses = self.head.forward_train(x, img_metas, gt_bboxes, gt_labels)
        return losses

    def forward_test(self, imgs, img_metas):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for img, img_meta in zip(imgs, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(img.size()[-2:])
        # print(imgs[0], img_metas)
        x = self.extract_feat(imgs[0])
        results = self.head.forward_test(x, img_metas[0])

        return results

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
