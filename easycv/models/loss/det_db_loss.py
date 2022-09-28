# Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/losses/det_db_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.models.builder import LOSSES


class BalanceLoss(nn.Module):

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 negative_ratio=3,
                 return_origin=False,
                 eps=1e-6,
                 **kwargs):
        """
               The BalanceLoss for Differentiable Binarization text detection
               args:
                   balance_loss (bool): whether balance loss or not, default is True
                   main_loss_type (str): can only be one of ['CrossEntropy','DiceLoss',
                       'Euclidean','BCELoss', 'MaskL1Loss'], default is  'DiceLoss'.
                   negative_ratio (int|float): float, default is 3.
                   return_origin (bool): whether return unbalanced loss or not, default is False.
                   eps (float): default is 1e-6.
               """
        super(BalanceLoss, self).__init__()
        self.balance_loss = balance_loss
        self.main_loss_type = main_loss_type
        self.negative_ratio = negative_ratio
        self.return_origin = return_origin
        self.eps = eps

        if self.main_loss_type == 'CrossEntropy':
            self.loss = nn.CrossEntropyLoss()
        elif self.main_loss_type == 'Euclidean':
            self.loss = nn.MSELoss()
        elif self.main_loss_type == 'DiceLoss':
            self.loss = DiceLoss(self.eps)
        elif self.main_loss_type == 'BCELoss':
            self.loss = BCELoss(reduction='none')
        elif self.main_loss_type == 'MaskL1Loss':
            self.loss = MaskL1Loss(self.eps)
        else:
            loss_type = [
                'CrossEntropy', 'DiceLoss', 'Euclidean', 'BCELoss',
                'MaskL1Loss'
            ]
            raise Exception(
                'main_loss_type in BalanceLoss() can only be one of {}'.format(
                    loss_type))

    def forward(self, pred, gt, mask=None):
        """
        The BalanceLoss for Differentiable Binarization text detection
        args:
            pred (variable): predicted feature maps.
            gt (variable): ground truth feature maps.
            mask (variable): masked maps.
        return: (variable) balanced loss
        """
        positive = gt * mask
        negative = (1 - gt) * mask

        positive_count = int(positive.sum())
        negative_count = int(
            min(negative.sum(), positive_count * self.negative_ratio))
        loss = self.loss(pred, gt, mask=mask)

        if not self.balance_loss:
            return loss

        positive_loss = positive * loss
        negative_loss = negative * loss
        negative_loss = torch.reshape(negative_loss, shape=[-1])
        if negative_count > 0:
            sort_loss, _ = negative_loss.sort(descending=True)
            negative_loss = sort_loss[:negative_count]
            # negative_loss, _ = paddle.topk(negative_loss, k=negative_count_int)
            balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
                positive_count + negative_count + self.eps)
        else:
            balance_loss = positive_loss.sum() / (positive_count + self.eps)
        if self.return_origin:
            return balance_loss, loss

        return balance_loss


class DiceLoss(nn.Module):
    '''
    Loss function from https://arxiv.org/abs/1707.03237,
    where iou computation is introduced heatmap manner to measure the
    diversity bwtween tow heatmaps.
    '''

    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask, weights=None):
        '''
        pred: one or two heatmaps of shape (N, 1, H, W),
            the losses of tow heatmaps are added together.
        gt: (N, 1, H, W)
        mask: (N, H, W)
        '''
        return self._compute(pred, gt, mask, weights)

    def _compute(self, pred, gt, mask, weights):
        if pred.dim() == 4:
            pred = pred[:, 0, :, :]
            gt = gt[:, 0, :, :]
        assert pred.shape == gt.shape
        assert pred.shape == mask.shape
        if weights is not None:
            assert weights.shape == mask.shape
            mask = weights * mask
        intersection = (pred * gt * mask).sum()

        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        loss = 1 - 2.0 * intersection / union
        assert loss <= 1
        return loss


class MaskL1Loss(nn.Module):

    def __init__(self, eps=1e-6):
        super(MaskL1Loss, self).__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt, mask):
        loss = (torch.abs(pred - gt) * mask).sum() / (mask.sum() + self.eps)
        return loss


class BCELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(BCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, label, mask=None, weight=None, name=None):
        loss = F.binary_cross_entropy(input, label, reduction=self.reduction)
        return loss


@LOSSES.register_module()
class DBLoss(nn.Module):
    """
    Differentiable Binarization (DB) Loss Function
    args:
        parm (dict): the super paramter for DB Loss
    """

    def __init__(self,
                 balance_loss=True,
                 main_loss_type='DiceLoss',
                 alpha=5,
                 beta=10,
                 ohem_ratio=3,
                 eps=1e-6,
                 **kwargs):
        super(DBLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.bce_loss = BalanceLoss(
            balance_loss=balance_loss,
            main_loss_type=main_loss_type,
            negative_ratio=ohem_ratio)

    def forward(self, predicts, labels):
        predict_maps = predicts['maps']
        # label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[
        #     1:]
        label_threshold_map, label_threshold_mask, label_shrink_map, label_shrink_mask = labels[
            'threshold_map'], labels['threshold_mask'], labels[
                'shrink_map'], labels['shrink_mask']
        if len(label_threshold_map.shape) == 4:
            label_threshold_map = label_threshold_map.squeeze(1)
            label_threshold_mask = label_threshold_mask.squeeze(1)
            label_shrink_map = label_shrink_map.squeeze(1)
            label_shrink_mask = label_shrink_mask.squeeze(1)
        shrink_maps = predict_maps[:, 0, :, :]
        threshold_maps = predict_maps[:, 1, :, :]
        binary_maps = predict_maps[:, 2, :, :]

        loss_shrink_maps = self.bce_loss(shrink_maps, label_shrink_map,
                                         label_shrink_mask)
        loss_threshold_maps = self.l1_loss(threshold_maps, label_threshold_map,
                                           label_threshold_mask)
        loss_binary_maps = self.dice_loss(binary_maps, label_shrink_map,
                                          label_shrink_mask)
        loss_shrink_maps = self.alpha * loss_shrink_maps
        loss_threshold_maps = self.beta * loss_threshold_maps

        # loss_all = loss_shrink_maps + loss_threshold_maps \
        #            + loss_binary_maps
        losses = {
            'loss_shrink_maps': loss_shrink_maps,
            'loss_threshold_maps': loss_threshold_maps,
            'loss_binary_maps': loss_binary_maps
        }
        return losses
