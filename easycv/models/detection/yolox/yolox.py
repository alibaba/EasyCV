# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.models.detection.utils import postprocess
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


def init_yolo(M):
    for m in M.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03


@MODELS.register_module
class YOLOX(BaseModel):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """
    param_map = {
        'nano': [0.33, 0.25],
        'tiny': [0.33, 0.375],
        's': [0.33, 0.5],
        'm': [0.67, 0.75],
        'l': [1.0, 1.0],
        'x': [1.33, 1.25]
    }

    # TODO configs support more params
    # backbone(Darknet)、neck(YOLOXPAFPN)、head(YOLOXHead)
    def __init__(self,
                 model_type: str = 's',
                 num_classes: int = 80,
                 test_size: tuple = (640, 640),
                 test_conf: float = 0.01,
                 nms_thre: float = 0.65,
                 pretrained: str = None):
        super(YOLOX, self).__init__()
        assert model_type in self.param_map, f'invalid model_type for yolox {model_type}, valid ones are {list(self.param_map.keys())}'

        in_channels = [256, 512, 1024]
        depth = self.param_map[model_type][0]
        width = self.param_map[model_type][1]

        self.backbone = YOLOPAFPN(depth, width, in_channels=in_channels)
        self.head = YOLOXHead(num_classes, width, in_channels=in_channels)

        self.apply(init_yolo)  # init_yolo(self)
        self.head.initialize_biases(1e-2)

        self.num_classes = num_classes
        self.test_conf = test_conf
        self.nms_thre = nms_thre
        self.test_size = test_size

    def forward_train(self,
                      img: Tensor,
                      gt_bboxes: Tensor,
                      gt_labels: Tensor,
                      img_metas=None,
                      scale=None) -> Dict[str, Tensor]:
        """ Abstract interface for model forward in training

        Args:
            img (Tensor): image tensor, NxCxHxW
            target (List[Tensor]): list of target tensor, NTx5 [class,x_c,y_c,w,h]
        """

        # gt_bboxes = gt_bboxes.to(torch.float16)
        # gt_labels = gt_labels.to(torch.float16)

        fpn_outs = self.backbone(img)

        targets = torch.cat([gt_labels, gt_bboxes], dim=2)

        loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
            fpn_outs, targets, img)

        outputs = {
            'total_loss':
            loss,
            'iou_l':
            iou_loss,
            'conf_l':
            conf_loss,
            'cls_l':
            cls_loss,
            'img_h':
            torch.tensor(img_metas[0]['img_shape'][0],
                         device=loss.device).float(),
            'img_w':
            torch.tensor(img_metas[0]['img_shape'][1],
                         device=loss.device).float()
        }
        return outputs

    def forward_test(self, img: Tensor, img_metas=None) -> Tensor:
        """ Abstract interface for model forward in training

        Args:
            img (Tensor): image tensor, NxCxHxW
            target (List[Tensor]): list of target tensor, NTx5 [class,x_c,y_c,w,h]
        """
        with torch.no_grad():
            fpn_outs = self.backbone(img)
            outputs = self.head(fpn_outs)

            outputs = postprocess(outputs, self.num_classes, self.test_conf,
                                  self.nms_thre)

            detection_boxes = []
            detection_scores = []
            detection_classes = []
            img_metas_list = []

            for i in range(len(outputs)):
                if img_metas:
                    img_metas_list.append(img_metas[i])
                if outputs[i] is not None:
                    bboxes = outputs[i][:,
                                        0:4] if outputs[i] is not None else None
                    if img_metas:
                        bboxes /= img_metas[i]['scale_factor'][0]
                    detection_boxes.append(bboxes.cpu().numpy())
                    detection_scores.append(
                        (outputs[i][:, 4] * outputs[i][:, 5]).cpu().numpy())
                    detection_classes.append(
                        outputs[i][:, 6].cpu().numpy().astype(np.int32))
                else:
                    detection_boxes.append(None)
                    detection_scores.append(None)
                    detection_classes.append(None)

            test_outputs = {
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
                'img_metas': img_metas_list
            }

        return test_outputs

    def forward(self, img, mode='compression', **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)
        elif mode == 'compression':
            return self.forward_compression(img, **kwargs)

    def forward_compression(self, x):
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)
        outputs = self.head(fpn_outs)

        return outputs

    def forward_export(self, img):
        with torch.no_grad():
            fpn_outs = self.backbone(img)
            outputs = self.head(fpn_outs)

            outputs = postprocess(outputs, self.num_classes, self.test_conf,
                                  self.nms_thre)

        return outputs
