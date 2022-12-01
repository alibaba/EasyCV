# Copyright (c) 2022 IDEA. All Rights Reserved.
# Copyright (c) Alibaba, Inc. and its affiliates.
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.framework.errors import NotImplementedError
from easycv.models.builder import HEADS, build_neck
from easycv.models.detection.utils import (DetrPostProcess, box_xyxy_to_cxcywh,
                                           inverse_sigmoid)
from easycv.models.loss import CDNCriterion, HungarianMatcher, SetCriterion
from easycv.models.utils import MLP
from easycv.utils.dist_utils import get_dist_info, is_dist_available
from ..dab_detr.dab_detr_transformer import PositionEmbeddingSineHW
from .cdn_components import cdn_post_process, prepare_for_cdn


@HEADS.register_module()
class DINOHead(nn.Module):
    """ Initializes the DINO Head.
    See `paper: DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection
    <https://arxiv.org/abs/2203.03605>`_ for details.
    Parameters:
        backbone: torch module of the backbone to be used. See backbone.py
        transformer: torch module of the transformer architecture. See transformer.py
        num_classes: number of object classes
        num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                        Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
        aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.

        fix_refpoints_hw: -1(default): learn w and h for each box seperately
                            >0 : given fixed number
                            -2 : learn a shared w and h
    """

    def __init__(
            self,
            num_classes,
            embed_dims,
            in_channels=[512, 1024, 2048],
            query_dim=4,
            num_queries=300,
            num_select=300,
            random_refpoints_xy=False,
            num_patterns=0,
            dn_components=None,
            transformer=None,
            fix_refpoints_hw=-1,
            num_feature_levels=1,
            # two stage
            two_stage_type='standard',  # ['no', 'standard']
            two_stage_add_query_num=0,
            dec_pred_class_embed_share=True,
            dec_pred_bbox_embed_share=True,
            two_stage_class_embed_share=True,
            two_stage_bbox_embed_share=True,
            use_centerness=False,
            use_iouaware=False,
            losses_list=['labels', 'boxes'],
            decoder_sa_type='sa',
            temperatureH=20,
            temperatureW=20,
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

        super(DINOHead, self).__init__()

        self.matcher = HungarianMatcher(
            cost_dict=cost_dict, cost_class_type='focal_loss_cost')
        self.criterion = SetCriterion(
            num_classes,
            matcher=self.matcher,
            weight_dict=weight_dict,
            losses=losses_list,
            loss_class_type='focal_loss')
        if dn_components is not None:
            self.dn_criterion = CDNCriterion(
                num_classes,
                matcher=self.matcher,
                weight_dict=weight_dict,
                losses=losses_list,
                loss_class_type='focal_loss')
        self.postprocess = DetrPostProcess(
            num_select=num_select,
            use_centerness=use_centerness,
            use_iouaware=use_iouaware)
        self.transformer = build_neck(transformer)

        self.positional_encoding = PositionEmbeddingSineHW(
            embed_dims // 2,
            temperatureH=temperatureH,
            temperatureW=temperatureW,
            normalize=True)

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.query_dim = query_dim
        self.dn_components = dn_components

        self.random_refpoints_xy = random_refpoints_xy
        self.fix_refpoints_hw = fix_refpoints_hw

        # for dn training
        self.dn_number = self.dn_components['dn_number']
        self.dn_box_noise_scale = self.dn_components['dn_box_noise_scale']
        self.dn_label_noise_ratio = self.dn_components['dn_label_noise_ratio']
        self.dn_labelbook_size = self.dn_components['dn_labelbook_size']
        self.label_enc = nn.Embedding(self.dn_labelbook_size + 1, embed_dims)

        # prepare input projection layers
        self.num_feature_levels = num_feature_levels
        if num_feature_levels > 1:
            num_backbone_outs = len(in_channels)
            input_proj_list = []
            for i in range(num_backbone_outs):
                in_channels_i = in_channels[i]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels_i, embed_dims, kernel_size=1),
                        nn.GroupNorm(32, embed_dims),
                    ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels_i,
                            embed_dims,
                            kernel_size=3,
                            stride=2,
                            padding=1),
                        nn.GroupNorm(32, embed_dims),
                    ))
                in_channels_i = embed_dims
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            assert two_stage_type == 'no', 'two_stage_type should be no if num_feature_levels=1 !!!'
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(in_channels[-1], embed_dims, kernel_size=1),
                    nn.GroupNorm(32, embed_dims),
                )
            ])

        # prepare pred layers
        self.dec_pred_class_embed_share = dec_pred_class_embed_share
        self.dec_pred_bbox_embed_share = dec_pred_bbox_embed_share
        # prepare class & box embed
        _class_embed = nn.Linear(embed_dims, num_classes)
        _bbox_embed = MLP(embed_dims, embed_dims, 4, 3)
        # init the two embed layers
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)

        # fcos centerness & iou-aware & tokenlabel
        self.use_centerness = use_centerness
        self.use_iouaware = use_iouaware
        if self.use_centerness:
            _center_embed = MLP(embed_dims, embed_dims, 1, 3)
        if self.use_iouaware:
            _iou_embed = MLP(embed_dims, embed_dims, 1, 3)

        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [
                _bbox_embed for i in range(transformer.num_decoder_layers)
            ]
            if self.use_centerness:
                center_embed_layerlist = [
                    _center_embed
                    for i in range(transformer.num_decoder_layers)
                ]
            if self.use_iouaware:
                iou_embed_layerlist = [
                    _iou_embed for i in range(transformer.num_decoder_layers)
                ]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed)
                for i in range(transformer.num_decoder_layers)
            ]
            if self.use_centerness:
                center_embed_layerlist = [
                    copy.deepcopy(_center_embed)
                    for i in range(transformer.num_decoder_layers)
                ]
            if self.use_iouaware:
                iou_embed_layerlist = [
                    copy.deepcopy(_iou_embed)
                    for i in range(transformer.num_decoder_layers)
                ]

        if dec_pred_class_embed_share:
            class_embed_layerlist = [
                _class_embed for i in range(transformer.num_decoder_layers)
            ]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed)
                for i in range(transformer.num_decoder_layers)
            ]
        self.bbox_embed = nn.ModuleList(box_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        self.transformer.decoder.bbox_embed = self.bbox_embed
        self.transformer.decoder.class_embed = self.class_embed

        if self.use_centerness:
            self.center_embed = nn.ModuleList(center_embed_layerlist)
            self.transformer.decoder.center_embed = self.center_embed
        if self.use_iouaware:
            self.iou_embed = nn.ModuleList(iou_embed_layerlist)
            self.transformer.decoder.iou_embed = self.iou_embed

        # two stage
        self.two_stage_type = two_stage_type
        self.two_stage_add_query_num = two_stage_add_query_num
        assert two_stage_type in [
            'no', 'standard'
        ], 'unknown param {} of two_stage_type'.format(two_stage_type)
        if two_stage_type != 'no':
            if two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
                if self.use_centerness:
                    self.transformer.enc_out_center_embed = _center_embed
                if self.use_iouaware:
                    self.transformer.enc_out_iou_embed = _iou_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(
                    _bbox_embed)
                if self.use_centerness:
                    self.transformer.enc_out_center_embed = copy.deepcopy(
                        _center_embed)
                if self.use_iouaware:
                    self.transformer.enc_out_iou_embed = copy.deepcopy(
                        _iou_embed)

            if two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(
                    _class_embed)

            self.refpoint_embed = None
            if self.two_stage_add_query_num > 0:
                self.init_ref_points(two_stage_add_query_num)

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ['sa', 'ca_label', 'ca_content']
        # self.replace_sa_with_double_ca = replace_sa_with_double_ca
        if decoder_sa_type == 'ca_label':
            self.label_embedding = nn.Embedding(num_classes, embed_dims)
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = self.label_embedding
        else:
            for layer in self.transformer.decoder.layers:
                layer.label_embedding = None
            self.label_embedding = None

    def init_weights(self):
        # init input_proj
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, self.query_dim)

        if self.random_refpoints_xy:
            # import ipdb; ipdb.set_trace()
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        if self.fix_refpoints_hw > 0:
            print('fix_refpoints_hw: {}'.format(self.fix_refpoints_hw))
            assert self.random_refpoints_xy
            self.refpoint_embed.weight.data[:, 2:] = self.fix_refpoints_hw
            self.refpoint_embed.weight.data[:, 2:] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, 2:])
            self.refpoint_embed.weight.data[:, 2:].requires_grad = False
        elif int(self.fix_refpoints_hw) == -1:
            pass
        elif int(self.fix_refpoints_hw) == -2:
            print('learn a shared h and w')
            assert self.random_refpoints_xy
            self.refpoint_embed = nn.Embedding(use_num_queries, 2)
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2])
            self.refpoint_embed.weight.data[:, :2].requires_grad = False
            self.hw_embed = nn.Embedding(1, 1)
        else:
            raise NotImplementedError('Unknown fix_refpoints_hw {}'.format(
                self.fix_refpoints_hw))

    def prepare(self, features, targets=None, mode='train'):

        if self.dn_number > 0 and targets is not None:
            input_query_label, input_query_bbox, attn_mask, dn_meta =\
                prepare_for_cdn(dn_args=(targets, self.dn_number, self.dn_label_noise_ratio, self.dn_box_noise_scale), num_queries=self.num_queries, num_classes=self.num_classes,
                                hidden_dim=self.embed_dims, label_enc=self.label_enc)
        else:
            assert targets is None
            input_query_bbox = input_query_label = attn_mask = dn_meta = None

        return input_query_bbox, input_query_label, attn_mask, dn_meta

    def forward(self,
                feats,
                img_metas,
                query_embed=None,
                tgt=None,
                attn_mask=None,
                dn_meta=None):
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
        # construct binary masks which used for the transformer.
        # NOTE following the official DETR repo, non-zero values representing
        # ignored positions, while zero values means valid positions.
        bs = feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = feats[0].new_ones((bs, input_img_h, input_img_w))
        for img_id in range(bs):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        srcs = []
        masks = []
        poss = []
        for l, src in enumerate(feats):
            mask = F.interpolate(
                img_masks[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
            # position encoding
            pos_l = self.positional_encoding(mask)  # [bs, embed_dim, h, w]
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            poss.append(pos_l)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](feats[-1])
                else:
                    src = self.input_proj[l](srcs[-1])
                mask = F.interpolate(
                    img_masks[None].float(),
                    size=src.shape[-2:]).to(torch.bool)[0]
                # position encoding
                pos_l = self.positional_encoding(mask)  # [bs, embed_dim, h, w]
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        hs, reference, hs_enc, ref_enc, init_box_proposal = self.transformer(
            srcs, masks, query_embed, poss, tgt, attn_mask)
        # In case num object=0
        hs[0] += self.label_enc.weight[0, 0] * 0.0

        # deformable-detr-like anchor update
        # reference_before_sigmoid = inverse_sigmoid(reference[:-1]) # n_dec, bs, nq, 4
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
                zip(reference[:-1], self.bbox_embed, hs)):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(
                layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)

        # outputs_class = self.class_embed(hs)
        outputs_class = torch.stack([
            layer_cls_embed(layer_hs)
            for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
        ])

        outputs_center_list = None
        if self.use_centerness:
            outputs_center_list = torch.stack([
                layer_center_embed(layer_hs)
                for layer_center_embed, layer_hs in zip(self.center_embed, hs)
            ])

        outputs_iou_list = None
        if self.use_iouaware:
            outputs_iou_list = torch.stack([
                layer_iou_embed(layer_hs)
                for layer_iou_embed, layer_hs in zip(self.iou_embed, hs)
            ])

        reference = torch.stack(reference)[:-1][..., :2]
        if self.dn_number > 0 and dn_meta is not None:
            outputs_class, outputs_coord_list, outputs_center_list, outputs_iou_list, reference = cdn_post_process(
                outputs_class, outputs_coord_list, dn_meta, self._set_aux_loss,
                outputs_center_list, outputs_iou_list, reference)
        out = {
            'pred_logits':
            outputs_class[-1],
            'pred_boxes':
            outputs_coord_list[-1],
            'pred_centers':
            outputs_center_list[-1]
            if outputs_center_list is not None else None,
            'pred_ious':
            outputs_iou_list[-1] if outputs_iou_list is not None else None,
            'refpts':
            reference[-1],
        }

        out['aux_outputs'] = self._set_aux_loss(outputs_class,
                                                outputs_coord_list,
                                                outputs_center_list,
                                                outputs_iou_list, reference)

        # for encoder output
        if hs_enc is not None:
            # prepare intermediate outputs
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            if self.use_centerness:
                interm_center = self.transformer.enc_out_center_embed(
                    hs_enc[-1])
            if self.use_iouaware:
                interm_iou = self.transformer.enc_out_iou_embed(hs_enc[-1])
            out['interm_outputs'] = {
                'pred_logits': interm_class,
                'pred_boxes': interm_coord,
                'pred_centers': interm_center if self.use_centerness else None,
                'pred_ious': interm_iou if self.use_iouaware else None,
                'refpts': init_box_proposal[..., :2],
            }

        out['dn_meta'] = dn_meta

        return out

    @torch.jit.unused
    def _set_aux_loss(self,
                      outputs_class,
                      outputs_coord,
                      outputs_center=None,
                      outputs_iou=None,
                      reference=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{
            'pred_logits':
            a,
            'pred_boxes':
            b,
            'pred_centers':
            outputs_center[i] if outputs_center is not None else None,
            'pred_ious':
            outputs_iou[i] if outputs_iou is not None else None,
            'refpts':
            reference[i],
        } for i, (a,
                  b) in enumerate(zip(outputs_class[:-1], outputs_coord[:-1]))]

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

        query_embed, tgt, attn_mask, dn_meta = self.prepare(
            x, targets=targets, mode='train')

        outputs = self.forward(
            x,
            img_metas,
            query_embed=query_embed,
            tgt=tgt,
            attn_mask=attn_mask,
            dn_meta=dn_meta)

        # Avoid inconsistent num_boxes for set_critertion and dn_critertion
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_available():
            torch.distributed.all_reduce(num_boxes)
        _, world_size = get_dist_info()
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        losses = self.criterion(outputs, targets, num_boxes=num_boxes)
        losses.update(
            self.dn_criterion(outputs, targets, len(outputs['aux_outputs']),
                              num_boxes))

        return losses

    def forward_test(self, x, img_metas):
        query_embed, tgt, attn_mask, dn_meta = self.prepare(x, mode='test')

        outputs = self.forward(
            x,
            img_metas,
            query_embed=query_embed,
            tgt=tgt,
            attn_mask=attn_mask,
            dn_meta=dn_meta)

        ori_shape_list = []
        for i in range(len(img_metas)):
            ori_h, ori_w, _ = img_metas[i]['ori_shape']
            ori_shape_list.append(torch.as_tensor([ori_h, ori_w]))
        orig_target_sizes = torch.stack(ori_shape_list, dim=0)

        results = self.postprocess(outputs, orig_target_sizes, img_metas)
        return results
