# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.models.ocr.postprocess.db_postprocess import DBPostProcess
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger


@MODELS.register_module()
class DBNet(BaseModel):
    """DBNet for text detection
    """

    def __init__(
        self,
        backbone,
        neck,
        head,
        postprocess,
        loss=None,
        pretrained=None,
        **kwargs,
    ):
        super(DBNet, self).__init__()

        self.pretrained = pretrained

        self.backbone = builder.build_backbone(backbone)
        self.neck = builder.build_neck(neck)
        self.head = builder.build_head(head)
        self.loss = builder.build_loss(loss) if loss else None
        self.postprocess_op = DBPostProcess(**postprocess)
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
        x = self.backbone(x)
        # y["backbone_out"] = x
        x = self.neck(x)
        # y["neck_out"] = x
        x = self.head(x)
        return x

    def forward_train(self, img, **kwargs):
        predicts = self.extract_feat(img)
        loss = self.loss(predicts, kwargs)
        return loss

    def forward_test(self, img, **kwargs):
        shape_list = [
            img_meta['ori_img_shape'] for img_meta in kwargs['img_metas']
        ]
        with torch.no_grad():
            preds = self.extract_feat(img)
        post_results = self.postprocess_op(preds, shape_list)
        if 'ignore_tags' in kwargs['img_metas'][0]:
            ignore_tags = [
                img_meta['ignore_tags'] for img_meta in kwargs['img_metas']
            ]
            post_results['ignore_tags'] = ignore_tags
        if 'polys' in kwargs['img_metas'][0]:
            polys = [img_meta['polys'] for img_meta in kwargs['img_metas']]
            post_results['polys'] = polys
        return post_results

    def postprocess(self, preds, shape_list):

        post_results = self.postprocess_op(preds, shape_list)
        points_results = post_results['points']
        dt_boxes = []
        for idx in range(len(points_results)):
            dt_box = points_results[idx]
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

        rect = np.array([tl, tr, br, bl], dtype='float32')
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points
