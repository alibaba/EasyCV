# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List

import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init

from easycv.core.evaluation.metrics import accuracy
from easycv.utils.logger import get_root_logger
from easycv.utils.registry import build_from_cfg
from ..registry import HEADS, LOSSES


@HEADS.register_module
class ClsHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
        Should Notice Evtorch module design input always be feature_list = [tensor, tensor,...]
    """

    def __init__(self,
                 with_avg_pool=False,
                 label_smooth=0.0,
                 in_channels=2048,
                 with_fc=True,
                 num_classes=1000,
                 loss_config={
                     'type': 'CrossEntropyLossWithLabelSmooth',
                 },
                 input_feature_index=[0]):

        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.label_smooth = label_smooth
        self.with_fc = with_fc
        self.input_feature_index = input_feature_index

        logger = get_root_logger()

        if label_smooth > 0:
            assert isinstance(self.label_smooth, float) and 0 <= self.label_smooth <= 1, \
                'label_smooth must be given as a float number in [0,1]'
            logger.info(f'=> Augment: using label smooth={self.label_smooth}')
            loss_config['label_smooth'] = label_smooth
        loss_config['num_classes'] = num_classes

        self.criterion = build_from_cfg(loss_config, LOSSES)

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.with_fc:
            self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self,
                     pretrained=None,
                     init_linear='normal',
                     std=0.01,
                     bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            'Undefined init_linear: {}'.format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:

        x1 = x[self.input_feature_index[0]]
        if self.with_avg_pool and x1.dim() > 2:
            assert x1.dim() == 4, \
                'Tensor must has 4 dims, got: {}'.format(x1.dim())
            x1 = self.avg_pool(x1)

        x1 = x1.view(x1.size(0), -1)

        if self.with_fc:
            cls_score = self.fc_cls(x1)
        else:
            cls_score = x1
        return [cls_score]

    def loss(self, cls_score: List[torch.Tensor],
             labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            cls_score: [N x num_classes]
            labels: if don't use mixup, shape is [N],else [N x num_classes]
        """
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses['loss'] = self.criterion(cls_score[0], labels)
        if len(labels.shape) == 1:
            losses['acc'] = accuracy(cls_score[0], labels)
        return losses

    def mixup_loss(self, cls_score, labels_1, labels_2,
                   lam) -> Dict[str, torch.Tensor]:
        losses = dict()
        losses['loss'] = lam * self.criterion(cls_score[0], labels_1) + \
            (1 - lam) * self.criterion(cls_score[0], labels_2)
        return losses
