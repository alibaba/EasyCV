# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn

from easycv.models.builder import MODELS
from easycv.models.detection.yolox.yolo_head import YOLOXHead
from easycv.models.detection.yolox.yolo_pafpn import YOLOPAFPN
from easycv.models.detection.yolox.yolox import YOLOX


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
                 stage: str = 'EDGE',
                 model_type: str = 's',
                 num_classes: int = 80,
                 test_size: tuple = (640, 640),
                 test_conf: float = 0.01,
                 nms_thre: float = 0.65,
                 pretrained: str = None,
                 depth: float = 1.0,
                 width: float = 1.0,
                 max_model_params: float = 0.0,
                 max_model_flops: float = 0.0,
                 activation: str = 'silu',
                 in_channels: list = [256, 512, 1024],
                 backbone=None,
                 head=None):
        super(YOLOX_EDGE, self).__init__()

        if backbone is None:
            self.backbone = YOLOPAFPN(
                depth,
                width,
                in_channels=in_channels,
                depthwise=True,
                act=activation)
        if head is None:
            self.head = YOLOXHead(
                num_classes,
                width,
                in_channels=in_channels,
                depthwise=True,
                act=activation,
                stage=stage)

        self.apply(init_yolo)  # init_yolo(self)
        self.head.initialize_biases(1e-2)

        self.stage = stage
        self.num_classes = num_classes
        self.test_conf = test_conf
        self.nms_thre = nms_thre
        self.test_size = test_size
