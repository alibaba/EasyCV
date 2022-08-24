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
from easycv.models.ocr.loss.rec_multi_loss import MultiLoss



@MODELS.register_module(force=True)
class OCRRecNet(BaseModel):
    """for text recognition
    """
    def __init__(
        self,
        backbone,
        head,
        postprocess,
        neck=None,
        loss=None,
        pretrained=None,
        **kwargs,
    ):
        super(OCRRecNet, self).__init__()
        
        self.pretrained = pretrained
        
        self.backbone = eval(backbone.type)(**backbone)
        self.neck = builder.build_neck(neck) if neck else None
        self.head = builder.build_head(head)
        self.loss = eval(loss.type)(**loss) if loss else None
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
                    
    def extract_feat(self, x, label=None, valid_ratios=None):
        y = dict()
        x = self.backbone(x)
        y["backbone_out"] = x
        if self.neck:
            x = self.neck(x)
            y["neck_out"] = x
        x = self.head(x,label=label,valid_ratios=valid_ratios)
        # for multi head, save ctc neck out for udml
        if isinstance(x, dict) and 'ctc_nect' in x.keys():
            y['neck_out'] = x['ctc_neck']
            y['head_out'] = x
        elif isinstance(x, dict):
            y.update(x)
        else:
            y["head_out"] = x
        return y
    
    def forward_train(self, img, **kwargs):
        label_ctc = kwargs.get('label_ctc', None)
        label_sar = kwargs.get('label_sar', None)
        length = kwargs.get('length', None)
        valid_ratio = kwargs.get('valid_ratio', None)
        predicts = self.extract_feat(img, label=label_sar, valid_ratios=valid_ratio)
        loss = self.loss(predicts, label_ctc=label_ctc, label_sar=label_sar, length=length)
        # if loss['CTCLoss']==np.inf:
        #     print(kwargs['img_metas'])
        #     exit()
        return loss
    
    def forward_test(self, img, **kwargs):
        label_ctc = kwargs.get('label_ctc', None)
        result = {}
        with torch.no_grad():
            preds = self.extract_feat(img)
            if label_ctc==None:
                preds_text = self.postprocess(preds)
            else:
                preds_text, label_text = self.postprocess(preds,label_ctc)
                result['label_text'] = label_text
            result['preds_text']=preds_text
            return result
        
    def postprocess(self, preds, label=None):
        if isinstance(preds,dict):
            preds = preds['head_out']
        if isinstance(preds, list):
            preds = [v.cpu().detach().numpy() for v in preds]
        else:
            preds = preds.cpu().detach().numpy()
        label = label.cpu().detach().numpy() if label!=None else label
        text_out = self.postprocess_op(preds,label)

        return text_out

if __name__ == "__main__":
    from easycv.utils.config_tools import mmcv_config_fromfile
    from easycv.models import build_model
    cfg = mmcv_config_fromfile('configs/ocr/rec_model.py')
    model = build_model(cfg.model)
    print(model)
    