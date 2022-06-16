# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

import mmcv
import numpy as np
from easycv.models import builder
from easycv.models.base import BaseModel
from easycv.models.builder import MODELS
from easycv.utils.logger import get_root_logger, print_log
from easycv.utils.checkpoint import load_checkpoint
from mmcv.runner.hooks import HOOKS
HOOKS._module_dict.pop('YOLOXLrUpdaterHook', None)
from mmdet.core.mask import mask2bbox
from mmdet.core import encode_mask_results
from .matcher import HungarianMatcher
from .criterion import SetCriterion
from .panoptic_gt_processing import preprocess_gt

@MODELS.register_module()
class Mask2Former(BaseModel):

    def __init__(
        self,
        backbone,
        head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        ):
        super(Mask2Former, self).__init__()
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)
        self.head = builder.build_head(head)
        # building criterion
        self.num_classes = head.num_things_classes+head.num_stuff_classes
        self.num_things_classes = head.num_things_classes

        matcher = HungarianMatcher(
            cost_class=train_cfg.class_weight,
            cost_mask=train_cfg.mask_weight,
            cost_dice=train_cfg.dice_weight,
            num_points=train_cfg.num_points,
        )
        weight_dict = {"loss_ce": train_cfg.class_weight, "loss_mask": train_cfg.mask_weight, "loss_dice": train_cfg.dice_weight}

        if train_cfg.deep_supervision:
            dec_layers = train_cfg.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

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

    def forward_train(self, img, gt_labels,gt_masks,gt_semantic_seg,img_metas):
        features = self.backbone(img)
        outputs = self.head(features)
        targets = preprocess_gt(gt_labels,gt_masks,gt_semantic_seg,img_metas)
        losses = self.criterion(outputs, targets)
        return losses

    def forward_test(self, img,img_metas,rescale=True):
        features = self.backbone(img[0])
        outputs = self.head(features)
        mask_cls_results = outputs['pred_logits']
        mask_pred_results = outputs['pred_masks']

        results = []
        for mask_cls_result, mask_pred_result, meta in zip(
                mask_cls_results, mask_pred_results, img_metas[0]):
            # remove padding
            # print(meta)
            # img_height, img_width = meta['img_shape'][:2]
            # mask_pred_result = mask_pred_result[:, :img_height, :img_width]
            # print(mask_pred_result.shape)


            # if rescale:
            #     # return result in original resolution
            #     ori_height, ori_width = meta['ori_shape'][:2]
            #     mask_pred_result = F.interpolate(
            #         mask_pred_result[:, None],
            #         size=(ori_height, ori_width),
            #         mode='bilinear',
            #         align_corners=False)[:, 0]
            pad_height, pad_width = meta['pad_shape'][:2]
            mask_pred_result = F.interpolate(
                mask_pred_result[:, None],
                size=(pad_height, pad_width),
                mode='bilinear',
                align_corners=False)[:, 0]
            #remove padding
            img_height, img_width = meta['img_shape'][:2]
            mask_pred_result = mask_pred_result[:, :img_height, :img_width]
            ori_height, ori_width = meta['ori_shape'][:2]
            mask_pred_result = F.interpolate(
                mask_pred_result[:, None],
                size=(ori_height, ori_width),
                mode='bilinear',
                align_corners=False)[:, 0]

            result = dict()

            #instance_on
            ins_results = self.instance_postprocess(
                mask_cls_result, mask_pred_result)
            # result['ins_results'] = ins_results

            labels_per_image, bboxes, mask_pred_binary = ins_results
            
            labels_per_image, bboxes, mask_pred_binary = labels_per_image.cpu().numpy(),bboxes.cpu().numpy(),mask_pred_binary.cpu().numpy()
            mask_pred_binary = [mask_pred_binary]
            mask_pred_binary = encode_mask_results(mask_pred_binary)
            segms = []
            segms = mmcv.concat_list(mask_pred_binary)
            segms = np.stack(segms, axis=0)
            scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
            bboxes = bboxes[:, 0:4] if bboxes.shape[1] == 5 else bboxes
            detection_boxes = []
            detection_scores = []
            detection_classes = []
            detection_masks = []
            detection_boxes.append(bboxes)
            detection_scores.append(scores)
            detection_classes.append(labels_per_image)
            detection_masks.append(segms)
            assert len(img_metas) == 1
            outputs = {
                'detection_boxes': detection_boxes,
                'detection_scores': detection_scores,
                'detection_classes': detection_classes,
                'detection_masks': detection_masks,
                'img_metas': img_metas[0]
            }
        return outputs

    def forward(self, img, mode='train', gt_labels=None,gt_masks=None,gt_semantic_seg=None,img_metas=None,**kwargs):

        if mode == 'train':
            return self.forward_train(img, gt_labels,gt_masks,gt_semantic_seg,img_metas)
        elif mode == 'test':
            return self.forward_test(img,img_metas)
        else:
            raise Exception('No such mode: {}'.format(mode))

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
        max_per_image = self.test_cfg.get('max_per_image', 100)
        num_queries = mask_cls.shape[0]
        # shape (num_queries, num_class)
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # shape (num_queries * num_class, )
        labels = torch.arange(self.num_classes, device=mask_cls.device).\
            unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)
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

        return labels_per_image, bboxes, mask_pred_binary
        


            


    

        