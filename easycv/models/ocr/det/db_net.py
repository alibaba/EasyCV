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
from easycv.models.ocr.backbones.det_mobilenet_v3 import MobileNetV3
from easycv.models.ocr.postprocess.db_postprocess import DBPostProcess
from easycv.models.ocr.loss.det_db_loss import DBLoss


@MODELS.register_module()
class DBNet(BaseModel):
    """DBNet for text detection
    """
    def __init__(
        self,
        backbone,
        neck,
        head,
        loss,
        postprocess,
        pretrained=None,
        **kwargs,
    ):
        super(DBNet, self).__init__()
        
        self.pretrained = pretrained
        
        self.backbone = eval(backbone.type)(**backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.loss = eval(loss.type)(**loss)
        self.postprocess_op = DBPostProcess(**postprocess)
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
    
    def forward_train(self, img, **kwargs):
        predicts = self.extract_feat(img)
        loss = self.loss(predicts, kwargs)
        return loss
    
    def forward_test(self, img):
        with torch.no_grad():
            out = self.extract_feat(img)
            return out
    
    def postprocess(self, preds, shape_list):

        post_results = self.postprocess_op(preds, shape_list)
        dt_boxes = []
        for idx,post_result in enumerate(post_results):
            dt_box = post_result['points']
            dt_box = self.filter_tag_det_res(dt_box, shape_list[idx])
            dt_boxes.append(dt_box)
        return dt_boxes

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes
                    
    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

if __name__ == "__main__":
    from easycv.utils.config_tools import mmcv_config_fromfile
    from easycv.models import build_model
    cfg = mmcv_config_fromfile('configs/ocr/det_model.py')
    model = build_model(cfg.model)
    print(model)
    
    