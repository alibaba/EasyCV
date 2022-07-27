# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.detection.utils import (accuracy, box_cxcywh_to_xyxy,
                                           generalized_box_iou)
from easycv.models.loss.focal_loss import py_sigmoid_focal_loss
from easycv.models.utils import get_world_size, is_dist_avail_and_initialized


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self,
                 num_classes,
                 matcher,
                 weight_dict,
                 losses,
                 eos_coef=None,
                 loss_class_type='ce',
                 dn_components=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_class_type = loss_class_type
        if self.loss_class_type == 'ce':
            empty_weight = torch.ones(self.num_classes + 1)
            empty_weight[-1] = eos_coef
            self.register_buffer('empty_weight', empty_weight)
        if dn_components is not None:
            self.dn_criterion = DNCriterion(self.weight_dict)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t['labels'][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        target_classes[idx] = target_classes_o

        if self.loss_class_type == 'ce':
            loss_ce = F.cross_entropy(
                src_logits.transpose(1, 2), target_classes,
                self.empty_weight) * self.weight_dict['loss_ce']
        elif self.loss_class_type == 'focal_loss':
            target_classes_onehot = torch.zeros([
                src_logits.shape[0], src_logits.shape[1],
                src_logits.shape[2] + 1
            ],
                                                dtype=src_logits.dtype,
                                                layout=src_logits.layout,
                                                device=src_logits.device)
            target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
            target_classes_onehot = target_classes_onehot[:, :, :-1]

            loss_ce = py_sigmoid_focal_loss(
                src_logits,
                target_classes_onehot.long(),
                alpha=0.25,
                gamma=2,
                reduction='none').mean(1).sum() / num_boxes
            loss_ce = loss_ce * src_logits.shape[1] * self.weight_dict[
                'loss_ce']
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx],
                                                   target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v['labels']) for v in targets],
                                      device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) !=
                     pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat(
            [t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum(
        ) / num_boxes * self.weight_dict['loss_bbox']

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes),
                box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum(
        ) / num_boxes * self.weight_dict['loss_giou']

        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, mask_dict=None, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc

             return_indices: used for vis. if True, the layer0-5 indices will be returned as well.
        """

        outputs_without_aux = {
            k: v
            for k, v in outputs.items() if k != 'aux_outputs'
        }

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor([num_boxes],
                                    dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(
                self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices,
                                           num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if mask_dict is not None:
            # dn loss computation
            aux_num = 0
            if 'aux_outputs' in outputs:
                aux_num = len(outputs['aux_outputs'])
            dn_losses = self.dn_criterion(mask_dict, self.training, aux_num,
                                          0.25)
            losses.update(dn_losses)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class DNCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict

    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict[
            'output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice']

        known_indice = mask_dict['known_indice']

        batch_idx = mask_dict['batch_idx']
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(
                1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(
                1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt

    def tgt_loss_boxes(
        self,
        src_boxes,
        tgt_boxes,
        num_tgt,
    ):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if len(tgt_boxes) == 0:
            return {
                'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
                'tgt_loss_giou': torch.as_tensor(0.).to('cuda'),
            }

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

        losses = {}
        losses['tgt_loss_bbox'] = loss_bbox.sum(
        ) / num_tgt * self.weight_dict['loss_bbox']

        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(tgt_boxes)))
        losses['tgt_loss_giou'] = loss_giou.sum(
        ) / num_tgt * self.weight_dict['loss_giou']
        return losses

    def tgt_loss_labels(self,
                        src_logits_,
                        tgt_labels_,
                        num_tgt,
                        focal_alpha,
                        log=False):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        if len(tgt_labels_) == 0:
            return {
                'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
                'tgt_class_error': torch.as_tensor(0.).to('cuda'),
            }

        src_logits, tgt_labels = src_logits_.unsqueeze(
            0), tgt_labels_.unsqueeze(0)

        target_classes_onehot = torch.zeros([
            src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1
        ],
                                            dtype=src_logits.dtype,
                                            layout=src_logits.layout,
                                            device=src_logits.device)
        target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = py_sigmoid_focal_loss(
            src_logits,
            target_classes_onehot.long(),
            alpha=focal_alpha,
            gamma=2,
            reduction='none').mean(1).sum(
            ) / num_tgt * src_logits.shape[1] * self.weight_dict['loss_ce']

        losses = {'tgt_loss_ce': loss_ce}
        if log:
            losses['tgt_class_error'] = 100 - accuracy(src_logits_,
                                                       tgt_labels_)[0]
        return losses

    def forward(self, mask_dict, training, aux_num, focal_alpha):
        """
        compute dn loss in criterion
        Args:
            mask_dict: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}
        if training and 'output_known_lbs_bboxes' in mask_dict:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = self.prepare_for_loss(
                mask_dict)
            losses.update(
                self.tgt_loss_labels(output_known_class[-1], known_labels,
                                     num_tgt, focal_alpha))
            losses.update(
                self.tgt_loss_boxes(output_known_coord[-1], known_bboxs,
                                    num_tgt))
        else:
            losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')

        if aux_num:
            for i in range(aux_num):
                # dn aux loss
                if training and 'output_known_lbs_bboxes' in mask_dict:
                    l_dict = self.tgt_loss_labels(output_known_class[i],
                                                  known_labels, num_tgt,
                                                  focal_alpha)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    l_dict = self.tgt_loss_boxes(output_known_coord[i],
                                                 known_bboxs, num_tgt)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
