# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.
import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from easycv.models.base import BaseModel
from easycv.models.builder import MODELS, build_head
from easycv.models.detection.utils import postprocess
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

    def __init__(self,
                 model_type='s',
                 test_conf=0.01,
                 nms_thre=0.65,
                 backbone='CSPDarknet',
                 use_att=None,
                 asff_channel=2,
                 neck_type='yolo',
                 neck_mode='all',
                 num_classes=None,
                 head=None,
                 pretrained=True):
        super(YOLOX, self).__init__()

        assert model_type in self.param_map, f'invalid model_type for yolox {model_type}, valid ones are {list(self.param_map.keys())}'

        if num_classes is not None:
            # adapt to previous export model (before easycv0.6.0)
            logging.warning(
                'Warning: You are now attend to use an old YOLOX model before easycv0.6.0 with key num_classes'
            )
            head = dict(
                type='YOLOXHead',
                model_type=model_type,
                num_classes=num_classes,
            )

        # the change of backbone/neck/head only support model_type as 's'
        if model_type != 's':
            head_type = head.get('type', None)
            assert backbone == 'CSPDarknet' and neck_type == 'yolo' and neck_mode == 'all' and head_type == 'YOLOXHead', 'We only support the architecture modification for YOLOX-S.'

        self.pretrained = pretrained

        in_channels = [256, 512, 1024]
        depth = self.param_map[model_type][0]
        width = self.param_map[model_type][1]

        self.backbone = YOLOPAFPN(
            depth,
            width,
            backbone=backbone,
            neck_type=neck_type,
            neck_mode=neck_mode,
            in_channels=in_channels,
            asff_channel=asff_channel,
            use_att=use_att)

        if head is not None:
            # head is None for YOLOX-edge to define a special head
            # set and check model type in head as the same of yolox
            head_model_type = head.get('model_type', None)
            if head_model_type is None:
                head['model_type'] = model_type
            else:
                assert model_type == head_model_type, 'Please provide the same model_type of YOLOX in config.'
            self.head = build_head(head)
            self.num_classes = self.head.num_classes

        self.apply(init_yolo)  # init_yolo(self)
        self.test_conf = test_conf
        self.nms_thre = nms_thre
        self.use_trt_efficientnms = False  # TRT NMS only will be convert during export
        self.trt_efficientnms = None
        self.export_type = 'raw'  # export type will be convert during export

    def get_nmsboxes_num(self, img_scale=(640, 640)):
        """ Detection neck or head should provide nms box count information
        """
        if getattr(self, 'neck', None) is not None:
            return self.neck.get_nmsboxes_num(img_scale=(640, 640))
        else:
            return self.head.get_nmsboxes_num(img_scale=(640, 640))

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

            if self.head.decode_in_inference:
                if self.use_trt_efficientnms:
                    if self.trt_efficientnms is not None:
                        outputs = self.trt_efficientnms.forward(outputs)
                    else:
                        logging.error(
                            'PAI-YOLOX : using trt_efficientnms set to be True, but model has not attr(trt_efficientnms)'
                        )
                else:
                    if self.export_type == 'jit':
                        outputs = postprocess(outputs, self.num_classes,
                                              self.test_conf, self.nms_thre)
                    else:
                        logging.warning(
                            'PAI-YOLOX : export Blade model is not allowed to wrap the postprocess into jit script model'
                        )

        return outputs
