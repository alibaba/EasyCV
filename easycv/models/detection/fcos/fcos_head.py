# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.builder import HEADS
from .fcos_outputs import FCOSOutputs


def compute_locations(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride, dtype=torch.float32, device=device)
    shifts_y = torch.arange(
        0, h * stride, step=stride, dtype=torch.float32, device=device)
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale


class ModuleListDial(nn.ModuleList):

    def __init__(self, modules=None):
        super(ModuleListDial, self).__init__(modules)
        self.cur_position = 0

    def forward(self, x):
        result = self[self.cur_position](x)
        self.cur_position += 1
        if self.cur_position >= len(self):
            self.cur_position = 0
        return result


@HEADS.register_module()
class FCOSHead(nn.Module):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
    """

    def __init__(
        self,
        num_classes,
        in_channels,
        fpn_strides,
        num_cls_convs,
        num_bbox_convs,
        num_share_convs,
        use_scale,
        fcos_outputs_config={
            'loss_alpha': 0.25,
            'loss_gamma': 2.0,
            'center_sample': True,
            'radius': 1.5,
            'pre_nms_thresh_train': 0.05,
            'pre_nms_topk_train': 1000,
            'post_nms_topk_train': 100,
            'loc_loss_type': 'giou',
            'pre_nms_thresh_test': 0.05,
            'pre_nms_topk_test': 1000,
            'post_nms_topk_test': 100,
            'nms_thresh': 0.6,
            'thresh_with_ctr': False,
            'box_quality': 'ctrness',
            'num_classes': 80,
            'strides': [8, 16, 32, 64, 128],
            'sizes_of_interest': [64, 128, 256, 512],
            'loss_normalizer_cls': 'fg',
            'loss_weight_cls': 1.0
        }):

        super(FCOSHead, self).__init__()

        self.num_classes = num_classes
        self.fpn_strides = fpn_strides
        head_configs = {
            'cls': num_cls_convs,
            'bbox': num_bbox_convs,
            'share': num_share_convs
        }
        self.num_levels = len(in_channels)

        assert len(
            set(in_channels)) == 1, 'Each level must have the same channel!'
        in_channels = in_channels[0]

        self.in_channels_to_top_module = in_channels

        for head in head_configs:
            tower = []
            num_convs = head_configs[head]
            for i in range(num_convs):
                tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True))
                tower.append(nn.GroupNorm(32, in_channels))
                tower.append(nn.ReLU())
            self.add_module('{}_tower'.format(head), nn.Sequential(*tower))

        self.cls_logits = nn.Conv2d(
            in_channels, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1)

        if use_scale:
            self.scales = nn.ModuleList(
                [Scale(init_value=1.0) for _ in range(self.num_levels)])
        else:
            self.scales = None

        self.fcos_outputs = FCOSOutputs(fcos_outputs_config)

    def init_weights(self):
        for modules in [
                self.cls_tower, self.bbox_tower, self.share_tower,
                self.cls_logits, self.bbox_pred, self.ctrness
        ]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = compute_locations(h, w,
                                                    self.fpn_strides[level],
                                                    feature.device)
            locations.append(locations_per_level)
        return locations

    def forward(self, x):
        logits = []
        bbox_reg = []
        ctrness = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
        return logits, bbox_reg, ctrness

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self, x, gt_bboxes, gt_labels, img_metas):
        """Forward function for training mode.
        Args:
            x (list[Tensor]): Features from backbone.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        locations = self.compute_locations(x)
        logits_pred, reg_pred, ctrness_pred = self.forward(x)

        gt_instances = []
        for i in range(len(img_metas)):
            gt_instances.append({
                'img_meta': img_metas[i],
                'gt_bboxes': gt_bboxes[i],
                'gt_labels': gt_labels[i]
            })

        _, losses = self.fcos_outputs.losses(logits_pred, reg_pred,
                                             ctrness_pred, locations,
                                             gt_instances, [])

        return losses

    def forward_test(self, x, img_metas):
        locations = self.compute_locations(x)
        logits_pred, reg_pred, ctrness_pred = self.forward(x)

        results = self.fcos_outputs.predict_proposals(logits_pred, reg_pred,
                                                      ctrness_pred, locations,
                                                      x.image_sizes, [])
        return results
