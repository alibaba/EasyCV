# Copyright (c) Alibaba, Inc. and its affiliates.
import mmcv
import numpy as np
import torch
import torch.nn.functional as F

from easycv.core.utils.misc import multi_apply
from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.models.segmentation.utils.criterion import SetCriterion
from easycv.models.segmentation.utils.matcher import MaskHungarianMatcher
from easycv.models.segmentation.utils.panoptic_gt_processing import (
    preprocess_panoptic_gt, preprocess_semantic_gt)
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger, print_log

INSTANCE_OFFSET = 1000


@MODELS.register_module()
class Mask2Former(BaseModel):

    def __init__(
        self,
        backbone,
        head,
        train_cfg,
        test_cfg,
        pretrained=None,
    ):
        """Mask2Former Model

        Args:
            backbone (dict): config to build backbone
            head (dict): config to builg mask2former head
            train_cfg (dict): config of training strategy.
            test_cfg (dict): config of test strategy.
            pretrained (str, optional): path of model weights. Defaults to None.
        """
        super(Mask2Former, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.instance_on = test_cfg.get('instance_on', False)
        self.panoptic_on = test_cfg.get('panoptic_on', False)
        self.semantic_on = test_cfg.get('semantic_on', False)
        self.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)
        # building criterion
        self.num_classes = head.num_things_classes + head.num_stuff_classes
        self.num_things_classes = head.num_things_classes
        self.num_stuff_classes = head.num_stuff_classes

        matcher = MaskHungarianMatcher(
            cost_class=train_cfg.class_weight,
            cost_mask=train_cfg.mask_weight,
            cost_dice=train_cfg.dice_weight,
            num_points=train_cfg.num_points,
        )
        weight_dict = {
            'loss_ce': train_cfg.class_weight,
            'loss_mask': train_cfg.mask_weight,
            'loss_dice': train_cfg.dice_weight
        }

        if train_cfg.deep_supervision:
            dec_layers = train_cfg.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update(
                    {k + f'_{i}': v
                     for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ['labels', 'masks']
        self.criterion = SetCriterion(
            self.head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=train_cfg.no_object_weight,
            losses=losses,
            num_points=train_cfg.num_points,
            oversample_ratio=train_cfg.oversample_ratio,
            importance_sample_ratio=train_cfg.importance_sample_ratio,
        )

        self.init_weights()

    def init_weights(self):
        logger = get_root_logger()
        if isinstance(self.pretrained, str):
            load_checkpoint(
                self.backbone, self.pretrained, strict=False, logger=logger)
        elif self.pretrained:
            if self.backbone.__class__.__name__ == 'PytorchImageModelWrapper':
                self.backbone.init_weights(pretrained=self.pretrained)
            elif hasattr(self.backbone, 'default_pretrained_model_path'
                         ) and self.backbone.default_pretrained_model_path:
                print_log(
                    'load model from default path: {}'.format(
                        self.backbone.default_pretrained_model_path), logger)
                load_checkpoint(
                    self.backbone,
                    self.backbone.default_pretrained_model_path,
                    strict=False,
                    logger=logger)
            else:
                print_log('load model from init weights')
                self.backbone.init_weights()
        else:
            print_log('load model from init weights')
            self.backbone.init_weights()

    def forward_train(self,
                      img,
                      gt_labels=None,
                      gt_masks=None,
                      gt_semantic_seg=None,
                      img_metas=None,
                      **kwargs):
        features = self.backbone(img)
        outputs = self.head(features)
        if gt_labels != None:
            targets = self.preprocess_gt(gt_labels, gt_masks, gt_semantic_seg,
                                         img_metas)
        else:
            targets = self.preprocess_gt_semantic(gt_semantic_seg)
        losses = self.criterion(outputs, targets)
        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses

    def forward_test(self,
                     img,
                     img_metas,
                     rescale=True,
                     encode=True,
                     **kwargs):
        features = self.backbone(img[0])
        outputs = self.head(features)
        mask_cls_results = outputs['pred_logits']
        mask_pred_results = outputs['pred_masks']
        detection_boxes = []
        detection_scores = []
        detection_classes = []
        detection_masks = []
        pan_masks = []
        seg_pred = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas[0]):
            pad_height, pad_width = meta['pad_shape'][:2]
            mask_pred_result = F.interpolate(
                mask_pred_result[:, None],
                size=(pad_height, pad_width),
                mode='bilinear',
                align_corners=False)[:, 0]
            # remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]
            ori_height, ori_width = meta['ori_shape'][:2]
            mask_pred_result = F.interpolate(
                mask_pred_result[:, None],
                size=(ori_height, ori_width),
                mode='bilinear',
                align_corners=False)[:, 0]

            # instance_on
            if self.instance_on:
                from easycv.utils.mmlab_utils import encode_mask_results
                labels_per_image, bboxes, mask_pred_binary = self.instance_postprocess(
                    mask_cls_result, mask_pred_result)
                segms = []
                if mask_pred_binary is not None and labels_per_image.shape[
                        0] > 0:
                    mask_pred_binary = [mask_pred_binary]
                    if encode:
                        mask_pred_binary = encode_mask_results(
                            mask_pred_binary)
                    segms = mmcv.concat_list(mask_pred_binary)
                    segms = np.stack(segms, axis=0)
                scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
                bboxes = bboxes[:, 0:4] if bboxes.shape[1] == 5 else bboxes
                detection_boxes.append(bboxes)
                detection_scores.append(scores)
                detection_classes.append(labels_per_image)
                detection_masks.append(segms)
            # panoptic on
            if self.panoptic_on:
                pan_results = self.panoptic_postprocess(
                    mask_cls_result, mask_pred_result)
                pan_masks.append(pan_results.cpu().numpy())

            if self.semantic_on:
                mask_cls = F.softmax(mask_cls_result, dim=-1)[..., :-1]
                mask_pred = mask_pred_result.sigmoid()
                semseg = torch.einsum('qc,qhw->chw', mask_cls, mask_pred)
                semseg = semseg.argmax(dim=0).cpu().numpy()
                seg_pred.append(semseg)

        assert len(img_metas) == 1
        outputs = {'img_metas': img_metas[0]}
        if self.instance_on:
            outputs['detection_boxes'] = detection_boxes
            outputs['detection_scores'] = detection_scores
            outputs['detection_classes'] = detection_classes
            outputs['detection_masks'] = detection_masks
        if self.panoptic_on:
            outputs['pan_results'] = pan_masks
        if self.semantic_on:
            outputs['seg_pred'] = seg_pred
        return outputs

    def instance_postprocess(self, mask_cls, mask_pred):
        """Instance segmengation postprocess.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            tuple[Tensor]: Instance segmentation results.

            - labels_per_image (Tensor): Predicted labels,\
                shape (n, ).
            - bboxes (Tensor): Bboxes and scores with shape (n, 5) of \
                positive region in binary mask, the last column is scores.
            - mask_pred_binary (Tensor): Instance masks of \
                shape (n, h, w).
        """
        from easycv.utils.mmlab_utils import mask2bbox
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(
            self.num_classes,
            device=mask_cls.device).unsqueeze(0).repeat(num_queries,
                                                        1).flatten(0, 1)
        scores_per_image, top_indices = scores.flatten(0, 1).topk(
            max_per_image, sorted=False)
        labels_per_image = labels[top_indices]
        query_indices = top_indices // self.num_classes
        mask_pred = mask_pred[query_indices]

        # extract things
        is_thing = labels_per_image < self.num_things_classes
        scores_per_image = scores_per_image[is_thing]
        labels_per_image = labels_per_image[is_thing]
        mask_pred = mask_pred[is_thing]

        mask_pred_binary = (mask_pred > 0).float()
        mask_scores_per_image = (mask_pred.sigmoid() *
                                 mask_pred_binary).flatten(1).sum(1) / (
                                     mask_pred_binary.flatten(1).sum(1) + 1e-6)
        det_scores = scores_per_image * mask_scores_per_image
        mask_pred_binary = mask_pred_binary.bool()
        bboxes = mask2bbox(mask_pred_binary)
        bboxes = torch.cat([bboxes, det_scores[:, None]], dim=-1)

        labels_per_image = labels_per_image.detach().cpu().numpy()
        bboxes = bboxes.detach().cpu().numpy()
        mask_pred_binary = mask_pred_binary.detach().cpu().numpy()
        return labels_per_image, bboxes, mask_pred_binary

    def panoptic_postprocess(self, mask_cls, mask_pred):
        """Panoptic segmengation inference.

        Args:
            mask_cls (Tensor): Classfication outputs of shape
                (num_queries, cls_out_channels) for a image.
                Note `cls_out_channels` should includes
                background.
            mask_pred (Tensor): Mask outputs of shape
                (num_queries, h, w) for a image.

        Returns:
            Tensor: Panoptic segment result of shape \
                (h, w), each element in Tensor means: \
                ``segment_id = _cls + instance_id * INSTANCE_OFFSET``.
        """
        object_mask_thr = self.test_cfg.get('object_mask_thr', 0.8)
        iou_thr = self.test_cfg.get('iou_thr', 0.8)
        filter_low_score = self.test_cfg.get('filter_low_score', False)

        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        keep = labels.ne(self.num_classes) & (scores > object_mask_thr)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.full((h, w),
                                  self.num_classes,
                                  dtype=torch.int32,
                                  device=cur_masks.device)
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            pass
        else:
            cur_mask_ids = cur_prob_masks.argmax(0)
            instance_id = 1
            for k in range(cur_classes.shape[0]):
                pred_class = int(cur_classes[k].item())
                isthing = pred_class < self.num_things_classes
                mask = cur_mask_ids == k
                mask_area = mask.sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()

                if filter_low_score:
                    mask = mask & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0:
                    if mask_area / original_area < iou_thr:
                        continue

                    if not isthing:
                        # different stuff regions of same class will be
                        # merged here, and stuff share the instance_id 0.
                        panoptic_seg[mask] = pred_class
                    else:
                        panoptic_seg[mask] = (
                            pred_class + instance_id * INSTANCE_OFFSET)
                        instance_id += 1

        return panoptic_seg

    def preprocess_gt(self, gt_labels_list, gt_masks_list, gt_semantic_segs,
                      img_metas):
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID.
            target_shape (tuple[int]): Shape of output mask_preds.
                Resize the masks to shape of mask_preds.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        num_things_list = [self.num_things_classes] * len(gt_labels_list)
        num_stuff_list = [self.num_stuff_classes] * len(gt_labels_list)
        if gt_semantic_segs is None:
            gt_semantic_segs = [None] * len(gt_labels_list)

        targets = multi_apply(preprocess_panoptic_gt, gt_labels_list,
                              gt_masks_list, gt_semantic_segs, num_things_list,
                              num_stuff_list, img_metas)
        labels, masks = targets
        new_targets = []
        for label, mask in zip(labels, masks):
            new_targets.append({
                'labels': label,
                'masks': mask,
            })
        return new_targets

    def preprocess_gt_semantic(self, gt_semantic_segs):
        targets = multi_apply(preprocess_semantic_gt, gt_semantic_segs)
        labels, masks = targets
        new_targets = []
        for label, mask in zip(labels, masks):
            new_targets.append({
                'labels': label,
                'masks': mask,
            })
        return new_targets
