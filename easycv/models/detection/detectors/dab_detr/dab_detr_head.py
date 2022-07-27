# Copyright (c) 2022 IDEA. All Rights Reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import math

import numpy as np
import torch
import torch.nn as nn

from easycv.models.builder import HEADS, build_neck
from easycv.models.detection.utils import (HungarianMatcher, SetCriterion,
                                           box_cxcywh_to_xyxy,
                                           box_xyxy_to_cxcywh, inverse_sigmoid)
from easycv.models.utils import MLP
from .dn_components import dn_post_process, prepare_for_dn


@HEADS.register_module()
class DABDETRHead(nn.Module):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
    """

    def __init__(self,
                 num_classes,
                 embed_dims,
                 query_dim=4,
                 iter_update=True,
                 num_queries=300,
                 num_select=300,
                 random_refpoints_xy=False,
                 num_patterns=0,
                 bbox_embed_diff_each_layer=False,
                 dn_components=None,
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

        super(DABDETRHead, self).__init__()

        self.matcher = HungarianMatcher(
            cost_dict=cost_dict, cost_class_type='focal_loss_cost')
        self.criterion = SetCriterion(
            num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            losses=['labels', 'boxes'],
            loss_class_type='focal_loss',
            dn_components=dn_components)
        self.postprocess = PostProcess(num_select=num_select)
        self.transformer = build_neck(transformer)

        self.class_embed = nn.Linear(embed_dims, num_classes)
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList(
                [MLP(embed_dims, embed_dims, 4, 3) for i in range(6)])
        else:
            self.bbox_embed = MLP(embed_dims, embed_dims, 4, 3)
        if iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.query_dim = query_dim
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.dn_components = dn_components

        self.query_embed = nn.Embedding(num_queries, query_dim)
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            self.query_embed.weight.data[:, :2].uniform_(0, 1)
            self.query_embed.weight.data[:, :2] = inverse_sigmoid(
                self.query_embed.weight.data[:, :2])
            self.query_embed.weight.data[:, :2].requires_grad = False

        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning('num_patterns should be int but {}'.format(
                type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, embed_dims)

        if self.dn_components:
            # leave one dim for indicator
            self.label_enc = nn.Embedding(num_classes + 1, embed_dims - 1)

    def init_weights(self):
        self.transformer.init_weights()

        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value

        # import ipdb; ipdb.set_trace()
        # init bbox_embed
        if self.bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def prepare(self, feats, targets=None, mode='train'):
        bs = feats[0].shape[0]
        query_embed = self.query_embed.weight
        if self.dn_components:
            # default pipeline
            self.dn_components['num_patterns'] = self.num_patterns
            self.dn_components['targets'] = targets
            # prepare for dn
            tgt, query_embed, attn_mask, mask_dict = prepare_for_dn(
                mode, self.dn_components, query_embed, bs, self.num_queries,
                self.num_classes, self.embed_dims, self.label_enc)
            if self.num_patterns > 0:
                l = tgt.shape[0]
                tgt[l - self.num_queries * self.num_patterns:] += \
                    self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1)
            return query_embed, tgt, attn_mask, mask_dict
        else:
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            if self.num_patterns == 0:
                tgt = torch.zeros(
                    self.num_queries,
                    bs,
                    self.embed_dims,
                    device=query_embed.device)
            else:
                tgt = self.patterns.weight[:, None, None, :].repeat(
                    1, self.num_queries, bs,
                    1).flatten(0, 1)  # n_q*n_pat, bs, d_model
                query_embed = query_embed.repeat(self.num_patterns, 1,
                                                 1)  # n_q*n_pat, bs, d_model
        return query_embed, tgt, None, None

    def forward(self,
                feats,
                img_metas,
                query_embed=None,
                tgt=None,
                attn_mask=None,
                mask_dict=None):
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

        feats = self.transformer(
            feats, img_metas, query_embed, tgt, attn_mask=attn_mask)

        hs, reference = feats
        outputs_class = self.class_embed(hs)
        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference)
            tmp = self.bbox_embed(hs)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords)

        if mask_dict is not None:
            # dn post process
            outputs_class, outputs_coord = dn_post_process(
                outputs_class, outputs_coord, mask_dict)
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

        query_embed, tgt, attn_mask, mask_dict = self.prepare(
            x, targets=targets, mode='train')

        outputs = self.forward(
            x,
            img_metas,
            query_embed=query_embed,
            tgt=tgt,
            attn_mask=attn_mask,
            mask_dict=mask_dict)

        losses = self.criterion(outputs, targets, mask_dict)

        return losses

    def forward_test(self, x, img_metas):
        query_embed, tgt, attn_mask, mask_dict = self.prepare(x, mode='test')

        outputs = self.forward(
            x,
            img_metas,
            query_embed=query_embed,
            tgt=tgt,
            attn_mask=attn_mask,
            mask_dict=mask_dict)

        ori_shape_list = []
        for i in range(len(img_metas)):
            ori_h, ori_w, _ = img_metas[i]['ori_shape']
            ori_shape_list.append(torch.as_tensor([ori_h, ori_w]))
        orig_target_sizes = torch.stack(ori_shape_list, dim=0)

        results = self.postprocess(outputs, orig_target_sizes, img_metas)
        return results


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes, img_metas):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        boxes = box_cxcywh_to_xyxy(out_bbox)
        boxes = torch.gather(boxes, 1,
                             topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

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
