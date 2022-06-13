# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.builder import HEADS
from .fcos_outputs import FCOSOutputs


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
            'strides': [8, 16, 32, 64, 128]
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

    def forward(self, x, top_module=None, yield_bbox_towers=False):
        logits = []
        bbox_reg = []
        ctrness = []
        top_feats = []
        bbox_towers = []
        for l, feature in enumerate(x):
            feature = self.share_tower(feature)
            cls_tower = self.cls_tower(feature)
            bbox_tower = self.bbox_tower(feature)
            if yield_bbox_towers:
                bbox_towers.append(bbox_tower)

            logits.append(self.cls_logits(cls_tower))
            ctrness.append(self.ctrness(bbox_tower))
            reg = self.bbox_pred(bbox_tower)
            if self.scales is not None:
                reg = self.scales[l](reg)
            # Note that we use relu, as in the improved FCOS, instead of exp.
            bbox_reg.append(F.relu(reg))
            if top_module is not None:
                top_feats.append(top_module(bbox_tower))
        return logits, bbox_reg, ctrness, top_feats, bbox_towers

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
        outputs = self.forward(x, img_metas)

        # for i in range(len(img_metas)):
        #     img_h, img_w, _ = img_metas[i]['img_shape']
        #     # DETR regress the relative position of boxes (cxcywh) in the image.
        #     # Thus the learning target should be normalized by the image size, also
        #     # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
        #     factor = outputs['pred_boxes'].new_tensor(
        #         [img_w, img_h, img_w, img_h]).unsqueeze(0)
        #     gt_bboxes[i] = box_xyxy_to_cxcywh(gt_bboxes[i]) / factor

        targets = []
        for gt_label, gt_bbox in zip(gt_labels, gt_bboxes):
            targets.append({'labels': gt_label, 'boxes': gt_bbox})

        losses = self.criterion(outputs, targets)

        return losses

    def forward_test(self, x, img_metas):
        outputs = self.forward(x, img_metas)

        ori_shape_list = []
        for i in range(len(img_metas)):
            ori_h, ori_w, _ = img_metas[i]['ori_shape']
            ori_shape_list.append(torch.as_tensor([ori_h, ori_w]))
        orig_target_sizes = torch.stack(ori_shape_list, dim=0)

        results = self.postprocess(outputs, orig_target_sizes)
        return results
