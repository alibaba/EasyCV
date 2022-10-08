# Copyright (c) 2014-2021 Megvii Inc And Alibaba PAI-Teams. All rights reserved.
import logging
import math
from abc import abstractmethod
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.framework.errors import KeyError, RuntimeError
from easycv.models.backbones.network_blocks import BaseConv, DWConv
from easycv.models.backbones.repvgg_yolox_backbone import RepVGGBlock
from easycv.models.detection.utils import bboxes_iou
from easycv.models.loss import YOLOX_IOULoss


class YOLOXHead_Template(nn.Module):
    param_map = {
        'nano': [0.33, 0.25],
        'tiny': [0.33, 0.375],
        's': [0.33, 0.5],
        'm': [0.67, 0.75],
        'l': [1.0, 1.0],
        'x': [1.33, 1.25]
    }

    def __init__(self,
                 num_classes=80,
                 model_type='s',
                 strides=[8, 16, 32],
                 in_channels=[256, 512, 1024],
                 act='silu',
                 conv_type='conv',
                 stage='CLOUD',
                 obj_loss_type='BCE',
                 reg_loss_type='giou',
                 decode_in_inference=True,
                 width=None):
        """
        Args:
            num_classes (int): detection class numbers.
            width (float): model width. Default value: 1.0.
            strides (list): expanded strides. Default value: [8, 16, 32].
            in_channels (list): model conv channels set. Default value: [256, 512, 1024].
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Default value: False.
            stage (str): model stage, distinguish edge head to cloud head. Default value: CLOUD.
            obj_loss_type (str): the loss function of the obj conf. Default value: BCE.
            reg_loss_type (str): the loss function of the box prediction. Default value: giou.
        """
        super().__init__()
        if width is None and model_type in self.param_map:
            width = self.param_map[model_type][1]
        else:
            assert (width !=
                    None), 'Unknow model type must have a given width!'

        self.width = width
        self.n_anchors = 1
        self.num_classes = num_classes
        self.stage = stage
        self.decode_in_inference = decode_in_inference  # for deploy, set to False

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        default_conv_type_list = ['conv', 'dwconv', 'repconv']

        if conv_type not in default_conv_type_list:
            logging.warning(
                'YOLOX-PAI tood head conv_type must in [conv, dwconv, repconv], otherwise we use repconv as default'
            )
            conv_type = 'repconv'
        if conv_type == 'conv':
            Conv = BaseConv
        if conv_type == 'dwconv':
            Conv = DWConv
        if conv_type == 'repconv':
            Conv = RepVGGBlock

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
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction='none')

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction='none')

        self.iou_loss = YOLOX_IOULoss(
            reduction='none', loss_type=reg_loss_type)

        self.obj_loss_type = obj_loss_type
        if obj_loss_type == 'BCE':
            self.obj_loss = nn.BCEWithLogitsLoss(reduction='none')
        else:
            raise KeyError('Undefined loss type: {}'.format(obj_loss_type))

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

    def get_nmsboxes_num(self, img_scale=(640, 640)):
        """ Count all Yolox NMS box with img_scale and head stride config
        """
        assert (
            len(img_scale) == 2
        ), 'Export YoloX predictor config contains img_scale must be (int, int) tuple!'

        total_box_count = 0
        for stride in self.strides:
            total_box_count += (img_scale[0] / stride) * (
                img_scale[1] / stride)
        return total_box_count

    @abstractmethod
    def forward(self, xin, labels=None, imgs=None):
        pass

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

        # loss_iou1 = (self.iou_loss1(
        #     bbox_preds.view(-1, 4)[fg_masks], reg_targets,xyxy=False)).sum() / num_fg

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
        num_pos_anchors: int = fg_mask.sum().item(
        )  # number of positive anchors

        if num_pos_anchors == 0:
            gt_matched_classes = torch.zeros(0, device=fg_mask.device).long()
            pred_ious_this_matching = torch.rand(0, device=fg_mask.device)
            matched_gt_inds = gt_matched_classes
            num_fg = num_pos_anchors

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
