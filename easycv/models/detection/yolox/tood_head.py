# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.
import logging
import math
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.backbones.network_blocks import BaseConv, DWConv
from easycv.models.detection.utils import bboxes_iou
from easycv.models.loss import IOUloss
from easycv.models.loss import FocalLoss, VarifocalLoss
from mmcv.cnn import ConvModule, normal_init


class TaskDecomposition(nn.Module):
    """Task decomposition module in task-aligned predictor of TOOD.

    Args:
        feat_channels (int): Number of feature channels in TOOD head.
        stacked_convs (int): Number of conv layers in TOOD head.
        la_down_rate (int): Downsample rate of layer attention.
        conv_cfg (dict): Config dict for convolution layer.
        norm_cfg (dict): Config dict for normalization layer.
    """

    def __init__(self,
                 feat_channels,
                 stacked_convs,
                 la_down_rate=8,
                 conv_cfg=None,
                 norm_cfg=None):
        super(TaskDecomposition, self).__init__()
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.in_channels = self.feat_channels * self.stacked_convs
        self.norm_cfg = norm_cfg
        self.layer_attention = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels // la_down_rate, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.in_channels // la_down_rate,
                self.stacked_convs,
                1,
                padding=0), nn.Sigmoid())

        self.reduction_conv = ConvModule(
            self.in_channels,
            self.feat_channels,
            1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            bias=norm_cfg is None)

    def init_weights(self):
        for m in self.layer_attention.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001)
        normal_init(self.reduction_conv.conv, std=0.01)

    def forward(self, feat, avg_feat=None):
        b, c, h, w = feat.shape
        if avg_feat is None:
            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        weight = self.layer_attention(avg_feat)

        # here we first compute the product between layer attention weight and
        # conv weight, and then compute the convolution between new conv weight
        # and feature map, in order to save memory and FLOPs.
        conv_weight = weight.reshape(
            b, 1, self.stacked_convs,
            1) * self.reduction_conv.conv.weight.reshape(
            1, self.feat_channels, self.stacked_convs, self.feat_channels)
        conv_weight = conv_weight.reshape(b, self.feat_channels,
                                          self.in_channels)
        feat = feat.reshape(b, self.in_channels, h * w)
        feat = torch.bmm(conv_weight, feat).reshape(b, self.feat_channels, h,
                                                    w)
        if self.norm_cfg is not None:
            feat = self.reduction_conv.norm(feat)
        feat = self.reduction_conv.activate(feat)

        return feat


class TOODHead(nn.Module):

    def __init__(self,
                 num_classes,
                 width=1.0,
                 strides=[8, 16, 32],
                 in_channels=[256, 512, 1024],
                 act='silu',
                 depthwise=False,
                 stage='CLOUD',
                 obj_loss_type='l1',
                 reg_loss_type='iou',
                 stacked_convs=6,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 ):
        """
        Args:
            num_classes (int): detection class numbers.
            width (float): model width. Default value: 1.0.
            strides (list): expanded strides. Default value: [8, 16, 32].
            in_channels (list): model conv channels set. Default value: [256, 512, 1024].
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Default value: False.
            stage (str): model stage, distinguish edge head to cloud head. Default value: CLOUD.
            obj_loss_type (str): the loss function of the obj conf. Default value: l1.
            reg_loss_type (str): the loss function of the box prediction. Default value: l1.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.stage = stage
        self.decode_in_inference = True  # for deploy, set to False

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.feat_channels = int(256 * width)

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_decomps = nn.ModuleList()
        self.reg_decomps = nn.ModuleList()
        self.stems = nn.ModuleList()

        self.inter_convs = nn.ModuleList()

        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                ))
            self.cls_convs.append(
                nn.Sequential(*[
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                ]))
            self.reg_convs.append(
                nn.Sequential(*[
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                    Conv(
                        in_channels=int(256 * width),
                        out_channels=int(256 * width),
                        ksize=3,
                        stride=1,
                        act=act,
                    ),
                ]))
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                ))
            self.cls_decomps.append(
                TaskDecomposition(self.feat_channels,
                                  self.stacked_convs,
                                  self.stacked_convs * 8,
                                  self.conv_cfg, self.norm_cfg))
            self.reg_decomps.append(
                TaskDecomposition(self.feat_channels,
                                  self.stacked_convs,
                                  self.stacked_convs * 8,
                                  self.conv_cfg, self.norm_cfg)
            )

        for i in range(self.stacked_convs):
            conv_cfg = self.conv_cfg
            chn = self.feat_channels
            self.inter_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg))

        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction='none')

        self.iou_loss = IOUloss(reduction='none', loss_type=reg_loss_type)

        self.obj_loss_type = obj_loss_type
        if obj_loss_type == 'BCE':
            self.obj_loss = nn.BCEWithLogitsLoss(reduction='none')
        elif obj_loss_type == 'focal':
            self.obj_loss = FocalLoss(reduction='none')

        elif obj_loss_type == 'v_focal':
            self.obj_loss = VarifocalLoss(reduction='none')
        else:
            assert "Undefined loss type: {}".format(obj_loss_type)

        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_decomp, reg_decomp, cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_decomps, self.reg_decomps, self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)

            inter_feats = []
            for inter_conv in self.inter_convs:
                x = inter_conv(x)
                inter_feats.append(x)
            feat = torch.cat(inter_feats, 1)

            avg_feat = F.adaptive_avg_pool2d(feat, (1, 1))
            cls_x = cls_decomp(feat, avg_feat)
            reg_x = reg_decomp(feat, avg_feat)

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type())
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(
                        1, grid.shape[1]).fill_(stride_this_level).type_as(
                        xin[0]))
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(batch_size, self.n_anchors, 4,
                                                 hsize, wsize)
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4)
                    origin_preds.append(reg_output.clone())

            else:
                if self.stage == 'EDGE':
                    m = nn.Hardsigmoid()
                    output = torch.cat(
                        [reg_output, m(obj_output),
                         m(cls_output)], 1)
                else:
                    output = torch.cat([
                        reg_output,
                        obj_output.sigmoid(),
                        cls_output.sigmoid()
                    ], 1)

            outputs.append(output)

        if self.training:

            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )

        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat([x.flatten(start_dim=2) for x in outputs],
                                dim=2).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize,
                                                 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4,
                                2).reshape(batch_size,
                                           self.n_anchors * hsize * wsize, -1)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride, dtype=torch.int))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])

            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )

                except RuntimeError:
                    logging.error(
                        'OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size.')
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        'cpu',
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64),
                    self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        loss_iou = (self.iou_loss(
            bbox_preds.view(-1, 4)[fg_masks], reg_targets)).sum() / num_fg

        if self.obj_loss_type == 'focal':
            loss_obj = (
                           self.focal_loss(obj_preds.sigmoid().view(-1, 1), obj_targets)
                       ).sum() / num_fg
        else:
            loss_obj = (self.obj_loss(obj_preds.view(-1, 1),
                                      obj_targets)).sum() / num_fg
        loss_cls = (self.bcewithlog_loss(
            cls_preds.view(-1, self.num_classes)[fg_masks],
            cls_targets)).sum() / num_fg

        if self.use_l1:
            loss_l1 = (self.l1_loss(
                origin_preds.view(-1, 4)[fg_masks], l1_targets)).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def focal_loss(self, pred, gt):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.eq(0).float()
        pos_loss = torch.log(pred + 1e-5) * torch.pow(1 - pred, 2) * pos_inds * 0.75
        neg_loss = torch.log(1 - pred + 1e-5) * torch.pow(pred, 2) * neg_inds * 0.25
        loss = -(pos_loss + neg_loss)
        return loss

    def get_l1_target(self,
                      l1_target,
                      gt,
                      stride,
                      x_shifts,
                      y_shifts,
                      eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode='gpu',
    ):

        if mode == 'cpu':
            print('------------CPU Mode for This Batch-------------')
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )
        # reference to: https://github.com/Megvii-BaseDetection/YOLOX/pull/811
        # NOTE: Fix `selected index k out of range`
        npa: int = fg_mask.sum().item()  # number of positive anchors

        if npa == 0:
            gt_matched_classes = torch.zeros(0, device=fg_mask.device).long()
            pred_ious_this_matching = torch.rand(0, device=fg_mask.device)
            matched_gt_inds = gt_matched_classes
            num_fg = npa

            if mode == 'cpu':
                gt_matched_classes = gt_matched_classes.cuda()
                fg_mask = fg_mask.cuda()
                pred_ious_this_matching = pred_ious_this_matching.cuda()
                matched_gt_inds = matched_gt_inds.cuda()
                num_fg = num_fg.cuda()

            return (
                gt_matched_classes,
                fg_mask,
                pred_ious_this_matching,
                matched_gt_inds,
                num_fg,
            )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == 'cpu':
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image,
                                    bboxes_preds_per_image, False)

        if (torch.isnan(pair_wise_ious.max())):
            pair_wise_ious = bboxes_iou(gt_bboxes_per_image,
                                        bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64),
                      self.num_classes).float().unsqueeze(1).repeat(
                1, num_in_boxes_anchor, 1))
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == 'cpu':
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
            with torch.cuda.amp.autocast(enabled=False):
                cls_preds_ = (
                        cls_preds_.float().unsqueeze(0).repeat(num_gt, 1,
                                                               1).sigmoid_() *
                        obj_preds_.float().unsqueeze(0).repeat(num_gt, 1,
                                                               1).sigmoid_())
                pair_wise_cls_loss = F.binary_cross_entropy(
                    cls_preds_.sqrt_(), gt_cls_per_image,
                    reduction='none').sum(-1)
        else:
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1,
                                                           1).sigmoid_() *
                    obj_preds_.float().unsqueeze(0).repeat(num_gt, 1,
                                                           1).sigmoid_())
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction='none').sum(-1)

        del cls_preds_

        cost = (
                pair_wise_cls_loss + 3.0 * pair_wise_ious_loss + 100000.0 *
                (~is_in_boxes_and_center))

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt,
                                    fg_mask)

        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == 'cpu':
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image +
             0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image +
             0.5 * expanded_strides_per_image).unsqueeze(0).repeat(num_gt, 1))

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] -
             0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(
                1, total_num_anchors))
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] +
             0.5 * gt_bboxes_per_image[:, 2]).unsqueeze(1).repeat(
                1, total_num_anchors))
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] -
             0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(
                1, total_num_anchors))
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] +
             0.5 * gt_bboxes_per_image[:, 3]).unsqueeze(1).repeat(
                1, total_num_anchors))

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (
                                    gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (
                                    gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (
                                    gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (
                                    gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor]
                & is_in_centers[:, is_in_boxes_anchor])
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt,
                           fg_mask):

        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))

        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()

        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix *
                                   pair_wise_ious).sum(0)[fg_mask_inboxes]

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds
