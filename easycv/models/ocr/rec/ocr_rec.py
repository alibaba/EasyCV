# Copyright (c) Alibaba, Inc. and its affiliates.
#debug
import sys
sys.path.append('/root/code/ocr/EasyCV')

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger
from easycv.models.ocr.backbones.rec_mv1_enhance import MobileNetV1Enhance
from easycv.models.ocr.postprocess.rec_postprocess import CTCLabelDecode



@MODELS.register_module(force=True)
class OCRRecNet(BaseModel):
    """for text recognition
    """
    def __init__(
        self,
        backbone,
        neck,
        head,
        postprocess,
        pretrained=None,
        **kwargs,
    ):
        super(OCRRecNet, self).__init__()
        
        self.pretrained = pretrained
        
        self.backbone = eval(backbone.type)(**backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.postprocess_op = eval(postprocess.type)(**postprocess)
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
        y["backbone_out"] = x
        x = self.neck(x)
        y["neck_out"] = x
        x = self.head(x)
        # for multi head, save ctc neck out for udml
        if isinstance(x, dict) and 'ctc_nect' in x.keys():
            y['neck_out'] = x['ctc_neck']
            y['head_out'] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        return y
    
    def forward_train(self, img):
        pass
    
    def forward_test(self, img):
        with torch.no_grad():
            out = self.extract_feat(img)
            return out
        
    def postprocess(self, preds):

        if isinstance(preds,dict):
            preds = preds['head_out']
        if isinstance(preds, list):
            preds = [v.cpu().detach().numpy() for v in preds]
        else:
            preds = preds.cpu().detach().numpy()
        rec_result = self.postprocess_op(preds)
        return rec_result

if __name__ == "__main__":
    from easycv.utils.config_tools import mmcv_config_fromfile
    from easycv.models import build_model
    cfg = mmcv_config_fromfile('configs/ocr/rec_model.py')
    model = build_model(cfg.model)
    print(model)
    