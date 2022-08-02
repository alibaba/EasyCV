# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger


@MODELS.register_module()
class DBNet(BaseModel):
    """DBNet for text detection
    """
    def __init__(
        self,
        backbone,
        head,
        pretrained=None
    ):
        super(DBNet, self).__init__()
        
        self.pretrained = pretrained
        
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)
        
        self.init_weights()
        
    def init_weights(self):
        logger  = get_root_logger()
        if self.pretrained:
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        else:
            # weight initialization
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    

        