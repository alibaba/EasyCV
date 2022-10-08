# Copyright (c) Alibaba, Inc. and its affiliates.

import math
import warnings

import mmcv
import torch
import torch.nn as nn

from easycv.framework.errors import NotImplementedError
from easycv.models.detection.utils import bbox_overlaps
from easycv.models.loss.utils import weighted_loss
from ..registry import LOSSES


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss


@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def giou_loss(pred, target, eps=1e-7):
    r"""`Generalized Intersection over Union: A Metric and A Loss for Bounding
    Box Regression <https://arxiv.org/abs/1902.09630>`_.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).

    Return:
        Tensor: Loss tensor.
    """
    gious = bbox_overlaps(pred, target, mode='giou', is_aligned=True, eps=eps)
    loss = 1 - gious
    return loss


@LOSSES.register_module
class YOLOX_IOULoss(nn.Module):

    def __init__(self, reduction='none', loss_type='iou'):
        super(YOLOX_IOULoss, self).__init__()
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

        elif self.loss_type == 'siou':
            # angle cost
            c_h = torch.max(pred[:, 1], target[:, 1]) - torch.min(
                pred[:, 1], target[:, 1])
            c_w = torch.max(pred[:, 0], target[:, 0]) - torch.min(
                pred[:, 0], target[:, 0])
            sigma = torch.sqrt(((pred[:, :2] - target[:, :2])**2).sum(dim=1))

            angle_cost = 2 * (c_h * c_w) / (sigma**2)

            # distance cost
            gamma = 2 - angle_cost
            # gamma = 1
            c_dw = torch.max(pred[:, 0], target[:, 0]) - torch.min(
                pred[:, 0], target[:, 0]) + (pred[:, 2] + target[:, 2]) / 2
            c_dh = torch.max(pred[:, 1], target[:, 1]) - torch.min(
                pred[:, 1], target[:, 1]) + (pred[:, 3] + target[:, 3]) / 2
            p_x = ((target[:, 0] - pred[:, 0]) / c_dw)**2
            p_y = ((target[:, 1] - pred[:, 1]) / c_dh)**2
            dist_cost = 2 - torch.exp(-gamma * p_x) - torch.exp(-gamma * p_y)

            # shape cost
            theta = 4
            w_w = torch.abs(pred[:, 2] - target[:, 2]) / torch.max(
                pred[:, 2], target[:, 2])
            w_h = torch.abs(pred[:, 3] - target[:, 3]) / torch.max(
                pred[:, 3], target[:, 3])
            shape_cost = torch.pow((1 - torch.exp(-w_w)), theta) + torch.pow(
                (1 - torch.exp(-w_h)), theta)

            loss = 1 - iou + (dist_cost + shape_cost) / 2

        elif self.loss_type == 'giou':
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'diou':
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(
                c_br[:, 1] - c_tl[:, 1], 2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) +
                          torch.pow(pred[:, 1] - target[:, 1], 2)
                          )  # center diagonal squared

            diou = iou - (center_dis / convex_dis)
            loss = 1 - diou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'ciou':
            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(
                c_br[:, 1] - c_tl[:, 1], 2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) +
                          torch.pow(pred[:, 1] - target[:, 1], 2)
                          )  # center diagonal squared

            v = (4 / math.pi**2) * torch.pow(
                torch.atan(target[:, 2] / torch.clamp(target[:, 3], min=1e-7))
                - torch.atan(pred[:, 2] / torch.clamp(pred[:, 3], min=1e-7)),
                2)

            with torch.no_grad():
                alpha = v / ((1 + 1e-7) - iou + v)

            ciou = iou - (center_dis / convex_dis + alpha * v)

            loss = 1 - ciou.clamp(min=-1.0, max=1.0)

        elif self.loss_type == 'eiou':

            c_tl = torch.min((pred[:, :2] - pred[:, 2:] / 2),
                             (target[:, :2] - target[:, 2:] / 2))
            c_br = torch.max((pred[:, :2] + pred[:, 2:] / 2),
                             (target[:, :2] + target[:, 2:] / 2))
            convex_dis = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + torch.pow(
                c_br[:, 1] - c_tl[:, 1], 2) + 1e-7  # convex diagonal squared

            center_dis = (torch.pow(pred[:, 0] - target[:, 0], 2) +
                          torch.pow(pred[:, 1] - target[:, 1], 2)
                          )  # center diagonal squared

            dis_w = torch.pow(pred[:, 2] - target[:, 2], 2)
            dis_h = torch.pow(pred[:, 3] - target[:, 3], 2)

            C_w = torch.pow(c_br[:, 0] - c_tl[:, 0], 2) + 1e-7
            C_h = torch.pow(c_br[:, 1] - c_tl[:, 1], 2) + 1e-7

            eiou = iou - (center_dis / convex_dis) - (dis_w / C_w) - (
                dis_h / C_h)

            loss = 1 - eiou.clamp(min=-1.0, max=1.0)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


@LOSSES.register_module()
class IoULoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(IoULoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * iou_loss(
            pred,
            target,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss


@LOSSES.register_module()
class GIoULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(GIoULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # 0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # giou_loss of shape (n,)
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * giou_loss(
            pred,
            target,
            weight,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss
