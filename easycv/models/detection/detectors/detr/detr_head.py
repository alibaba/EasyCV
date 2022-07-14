# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.builder import HEADS, build_neck
from easycv.models.detection.utils import (HungarianMatcher, SetCriterion,
                                           box_cxcywh_to_xyxy,
                                           box_xyxy_to_cxcywh)
from easycv.models.utils import MLP


@HEADS.register_module()
class DETRHead(nn.Module):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
    """

    _version = 2

    def __init__(self,
                 num_classes,
                 embed_dims,
                 eos_coef=0.1,
                 transformer=None,
                 cost_dict={
                     'cost_class': 1,
                     'cost_bbox': 5,
                     'cost_giou': 2,
                 },
                 weight_dict={
                     'loss_ce': 1,
                     'loss_bbox': 5,
                     'loss_giou': 2
                 },
                 **kwargs):

        super(DETRHead, self).__init__()

        self.matcher = HungarianMatcher(cost_dict=cost_dict)
        self.criterion = SetCriterion(
            num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            eos_coef=eos_coef,
            losses=['labels', 'boxes'])
        self.postprocess = PostProcess()
        self.transformer = build_neck(transformer)

        self.class_embed = nn.Linear(embed_dims, num_classes + 1)
        self.bbox_embed = MLP(embed_dims, embed_dims, 4, 3)
        self.num_classes = num_classes

    def init_weights(self):
        """Initialize weights of the detr head."""
        self.transformer.init_weights()

    def forward(self, feats, img_metas):
        """Forward function.
        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            img_metas (list[dict]): List of image information.
        Returns:
            tuple[list[Tensor], list[Tensor]]: Outputs for all scale levels.
                - all_cls_scores_list (list[Tensor]): Classification scores \
                    for each scale level. Each is a 4D-tensor with shape \
                    [nb_dec, bs, num_query, cls_out_channels]. Note \
                    `cls_out_channels` should includes background.
                - all_bbox_preds_list (list[Tensor]): Sigmoid regression \
                    outputs for each scale level. Each is a 4D-tensor with \
                    normalized coordinate format (cx, cy, w, h) and shape \
                    [nb_dec, bs, num_query, 4].
        """
        feats = self.transformer(feats, img_metas)

        outputs_class = self.class_embed(feats)
        outputs_coord = self.bbox_embed(feats).sigmoid()

        out = {
            'pred_logits': outputs_class[-1],
            'pred_boxes': outputs_coord[-1]
        }

        out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            'pred_logits': a,
            'pred_boxes': b
        } for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]

    # over-write because img_metas are needed as inputs for bbox_head.
    def forward_train(self, x, img_metas, gt_bboxes, gt_labels):
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
        # prepare ground truth
        for i in range(len(img_metas)):
            img_h, img_w, _ = img_metas[i]['img_shape']
            # DETR regress the relative position of boxes (cxcywh) in the image.
            # Thus the learning target should be normalized by the image size, also
            # the box format should be converted from defaultly x1y1x2y2 to cxcywh.
            factor = gt_bboxes[i].new_tensor([img_w, img_h, img_w,
                                              img_h]).unsqueeze(0)
            gt_bboxes[i] = box_xyxy_to_cxcywh(gt_bboxes[i]) / factor

        targets = []
        for gt_label, gt_bbox in zip(gt_labels, gt_bboxes):
            targets.append({'labels': gt_label, 'boxes': gt_bbox})

        outputs = self.forward(x, img_metas)

        losses = self.criterion(outputs, targets)

        return losses

    def forward_test(self, x, img_metas):
        outputs = self.forward(x, img_metas)

        ori_shape_list = []
        for i in range(len(img_metas)):
            ori_h, ori_w, _ = img_metas[i]['ori_shape']
            ori_shape_list.append(torch.as_tensor([ori_h, ori_w]))
        orig_target_sizes = torch.stack(ori_shape_list, dim=0)

        results = self.postprocess(outputs, orig_target_sizes, img_metas)
        return results


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    @torch.no_grad()
    def forward(self, outputs, target_sizes, img_metas):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h],
                                dim=1).to(boxes.device)
        boxes = boxes * scale_fct[:, None, :]

        results = {
            'detection_boxes': [boxes[0].cpu().numpy()],
            'detection_scores': [scores[0].cpu().numpy()],
            'detection_classes': [labels[0].cpu().numpy().astype(np.int32)],
            'img_metas': img_metas
        }

        return results
