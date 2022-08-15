# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn

from easycv.models.builder import (MODELS, build_backbone, build_head,
                                   build_neck)
from easycv.models.detection.detectors.yolox.yolo_head import YOLOXHead
from easycv.models.detection.detectors.yolox.yolo_pafpn import YOLOPAFPN
from easycv.models.detection.detectors.yolox.yolox import YOLOX


def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


@MODELS.register_module
class YOLOX_EDGE(YOLOX):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self,
                 backbone,
                 test_conf,
                 nms_thre,
                 head=None,
                 neck=None,
                 pretrained=True):
        super(YOLOX, self).__init__()

        self.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        self.head = build_head(head)

        self.apply(init_yolo)  # init_yolo(self)
        self.num_classes = self.head.num_classes
        self.test_conf = test_conf
        self.nms_thre = nms_thre
