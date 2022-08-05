# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger
from .. import builder
from ..base import BaseModel
from ..registry import MODELS


@MODELS.register_module
class BYOL(BaseModel):
    '''BYOL unofficial implementation. Paper: https://arxiv.org/abs/2006.07733
    '''

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 pretrained=None,
                 base_momentum=0.996,
                 **kwargs):
        super(BYOL, self).__init__()

        self.pretrained = pretrained
        self.online_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.target_net = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.online_net[0]
        for param in self.target_net.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights()

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self.online_net[0],
                self.pretrained,
                strict=False,
                logger=logger)
        else:
            self.online_net[0].init_weights()
        self.online_net[1].init_weights(init_linear='kaiming')  # projection
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # init the predictor in the head
        self.head.init_weights()

    @torch.no_grad()
    def _momentum_update(self):
        """
        Momentum update of the target network.
        """
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                param_ol.data * (1. - self.momentum)

    def forward_train(self, img, **kwargs):
        assert isinstance(img, list)
        assert len(img) == 2
        for _img in img:
            assert _img.dim() == 4, \
                'Input must have 4 dims, got: {}'.format(_img.dim())

        img_v1 = img[0].contiguous()
        img_v2 = img[1].contiguous()

        # compute query features
        proj_online_v1 = self.online_net(img_v1)[0]
        proj_online_v2 = self.online_net(img_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.target_net(img_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(img_v2)[0].clone().detach()

        loss = self.head(proj_online_v1, proj_target_v2)['loss'] + \
            self.head(proj_online_v2, proj_target_v1)['loss']
        self._momentum_update()
        return dict(loss=loss)

    def forward_test(self, img, **kwargs):
        pass

    def forward(self, img, mode='train', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'extract':
            return self.backbone(img)
        else:
            raise Exception('No such mode: {}'.format(mode))
