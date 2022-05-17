# Copyright (c) Alibaba, Inc. and its affiliates.

import torch
import torch.nn as nn

from ..registry import LOSSES
import math

@LOSSES.register_module
class IOUloss(nn.Module):

    def __init__(self, reduction='none', loss_type='iou'):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]
        if target.dtype != pred.dtype:
            target = target.to(pred.dtype)
        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max((pred[:, :2] - pred[:, 2:] / 2),
                       (target[:, :2] - target[:, 2:] / 2))
        br = torch.min((pred[:, :2] + pred[:, 2:] / 2),
                       (target[:, :2] + target[:, 2:] / 2))

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == 'iou':
            loss = 1 - iou**2
        elif self.loss_type == 'giou':
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "diou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)  # 包围框的左上点
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)  # 包围框的右下点
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            diou = iou - (center_dis / convex_dis)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "ciou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            v = (4 / math.pi ** 2) * torch.pow(torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7)) -
                                               torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)), 2)

            with torch.no_grad():
                alpha = v / ((1 + 1e-7) - iou + v)

            ciou = iou - (center_dis / convex_dis + alpha * v)

            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == "eiou":

            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(c_br[:, 1] - c_tl[:, 1],
                                                                           2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) + torch.pow(pred[:, 1] - target[:, 1],
                                                                              2))  # center diagonal squared

            dis_w = torch.pow(pred[:, 2] - target[:, 2], 2)  # 两个框的w欧式距离
            dis_h = torch.pow(pred[:, 3] - target[:, 3], 2)  # 两个框的h欧式距离

            C_w = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + 1e-7  # 包围框的w平方
            C_h = torch.pow(c_br[:, 1] - c_tl[:, 1], 2) + 1e-7  # 包围框的h平方

            eiou = iou - (center_dis / convex_dis) - (dis_w / C_w) - (dis_h / C_h)

            loss = 1 - eiou.clamp(min=-1.0, max=1.0)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


