# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger


@MODELS.register_module()
class TextClassifier(BaseModel):
    """for text classification
    """

    def __init__(
        self,
        backbone,
        head,
        neck=None,
        loss=None,
        pretrained=None,
        **kwargs,
    ):
        super(TextClassifier, self).__init__()

        self.pretrained = pretrained

        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck) if neck else None
        self.head = builder.build_head(head)
        self.loss = nn.CrossEntropyLoss()
        self.init_weights()

    def init_weights(self):
        logger = get_root_logger()
        if self.pretrained:
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        else:
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(
                        m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def extract_feat(self, x):
        y = dict()
        x = self.backbone(x)
        y['backbone_out'] = x
        if self.neck:
            x = self.neck(x)
            y['neck_out'] = x
        # convert to list in order to fit easycv cls head
        x = self.head([x])[0]
        x = F.softmax(x, dim=1)
        y['head_out'] = x
        return y

    def forward_train(self, img, label, **kwargs):
        out = {}
        preds = self.extract_feat(img)
        out['loss'] = self.loss(preds['head_out'], label)
        return out

    def forward_test(self, img, **kwargs):
        label = kwargs.get('label', None)
        result = {}
        preds = self.extract_feat(img)
        if label != None:
            result['label'] = label.cpu()
        result['neck'] = preds['head_out'].cpu()
        result['class'] = torch.argmax(preds['head_out'], dim=1).cpu()
        return result
