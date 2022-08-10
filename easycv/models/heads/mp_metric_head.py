# Copyright (c) Alibaba, Inc. and its affiliates.
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import kaiming_init, normal_init
from mmcv.runner import get_dist_info
from pytorch_metric_learning.miners import *
from torch import Tensor

from easycv.core.evaluation.metrics import accuracy
from easycv.models.loss import CrossEntropyLossWithLabelSmooth
from easycv.models.utils import DistributedLossWrapper, DistributedMinerWrapper
from easycv.utils.logger import get_root_logger
from easycv.utils.registry import build_from_cfg
from ..registry import HEADS, LOSSES

# Softmax based loss doesn't need ddp, the big fc while slowdown the training process.
MP_NODDP_LOSS = set([
    'ArcFaceLoss', 'AngularLoss', 'CosFaceLoss', 'LargeMarginSoftmaxLoss',
    'NormalizedSoftmaxLoss', 'SphereFaceLoss',
    'CrossEntropyLossWithLabelSmooth', 'AMSoftmaxLoss'
])


def EmbeddingExplansion(embs, labels, explanion_rate=4, alpha=1.0):
    """
        Expand embedding: CVPR refer to  https://github.com/clovaai/embedding-expansion
        combine PK sampled data, mixup anchor positive pair to generate more features, always combine with BatchHardminer.
        result on SOP and CUB need to be add

        Args:
            embs: [N , dims] tensor
            labels: [N] tensor
            explanion_rate: to expand N to explanion_rate * N
            alpha: beta distribution parameter for mixup

        Return:
            embs: [N*explanion_rate , dims]
    """

    embs = embs[0]
    label_list = labels.cpu().data.numpy()
    label_keys = []
    label_idx = {}

    # caculate label and mixup
    old = -1
    for idx, i in enumerate(label_list):
        if i == old or i in label_idx.keys():
            label_idx[old].append(idx)
        else:
            label_idx[i] = [idx]
            old = i
            label_keys.append(old)

    res = embs
    res_label = labels

    for j in range(explanion_rate - 1):
        refine_label_list = []
        for key in label_keys:
            random.shuffle(label_idx[key])
            refine_label_list += label_idx[key]

        refine_label_list = torch.tensor(refine_label_list).to(embs.device)

        if alpha > 0:
            lam = np.random.beta(
                alpha, alpha, size=[refine_label_list.size(0)])
        else:
            lam = 1

        lam = torch.tensor(lam).view(len(refine_label_list),
                                     -1).to(embs.device)

        data_mixed = lam * embs + (1 - lam) * embs[refine_label_list, :]
        data_mixed = data_mixed.float()
        res = torch.cat([res, data_mixed])
        res_label = torch.cat([res_label, labels])

    return [res], res_label


@HEADS.register_module
class MpMetrixHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    """

    def __init__(
        self,
        with_avg_pool=False,
        in_channels=2048,
        loss_config=[{
            'type': 'CircleLoss',
            'loss_weight': 1.0,
            'norm': True,
            'ddp': True,
            'm': 0.4,
            'gamma': 80
        }],
        input_feature_index=[0],
        input_label_index=0,
        ignore_label=None,
    ):

        super(MpMetrixHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.input_feature_index = input_feature_index
        self.input_label_index = input_label_index
        self.ignore_label = ignore_label

        rank, world_size = get_dist_info()

        logger = get_root_logger()
        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.loss_list = []
        self.norm_list = []
        self.loss_weight_list = []
        self.ddp_list = []
        self.miner_list = []
        assert len(loss_config) > 0
        for idx, loss in enumerate(loss_config):
            self.loss_weight_list.append(loss.pop('loss_weight', 1.0))
            self.norm_list.append(loss.pop('norm', True))
            cbm_param = loss.pop('cbm', None)
            miner_param = loss.pop('miner', None)
            name = loss['type']

            # ddp will be True is user not set and name not in MP_NODDP_LOSS
            ddp = loss.pop('ddp', None)
            if ddp is None:
                if name in MP_NODDP_LOSS:
                    ddp = False
                else:
                    ddp = True
            self.ddp_list.append(ddp)

            if world_size > 1 and self.ddp_list[idx]:
                tmp = build_from_cfg(loss, LOSSES)
                tmp_loss = DistributedLossWrapper(loss=tmp)
            else:
                tmp_loss = build_from_cfg(loss, LOSSES)

            if miner_param is not None:
                name = miner_param.pop('type')
                if world_size > 1 and self.ddp_list[idx]:
                    self.miner_list.append(
                        DistributedMinerWrapper(eval(name)(**miner_param)))
                else:
                    self.miner_list.append(eval(name)(**miner_param))
            else:
                self.miner_list.append(None)

            setattr(self, '%s_%d' % (name, idx), tmp_loss)
            self.loss_list.append(getattr(self, '%s_%d' % (name, idx)))

    def init_weights(self,
                     pretrained=None,
                     init_linear='normal',
                     std=0.01,
                     bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            'Undefined init_linear: {}'.format(init_linear)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')
            elif isinstance(m,
                            (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: List[torch.Tensor]) -> List[torch.Tensor]:
        # multi head feature distribute
        for i in self.input_feature_index:
            assert i < len(x)
        x = [x[i] for i in self.input_feature_index]
        assert isinstance(x, (tuple, list)) and len(x) == 1

        x1 = x[0]
        if self.with_avg_pool and x1.dim() > 2:
            assert x1.dim() == 4, \
                'Tensor must has 4 dims, got: {}'.format(x.dim())
            x1 = self.avg_pool(x1)
        x1 = x1.view(x1.size(0), -1)

        if hasattr(self, 'fc_cls'):
            cls_score = self.fc_cls(x1)
        else:
            cls_score = x1

        return [cls_score]

    def loss(self, cls_score, labels) -> Dict[str, torch.Tensor]:
        logger = get_root_logger()

        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1

        if type(labels) == list:
            assert (self.input_label_index < len(labels))
            tlabel = labels[self.input_label_index]
        else:
            tlabel = labels

        if self.ignore_label is not None:
            ignore_mask = tlabel.eq(self.ignore_label)
            no_ignore_mask = ~ignore_mask
            tlabel = torch.masked_select(tlabel, no_ignore_mask)
            no_ignore_idx = torch.where(no_ignore_mask == True)[0]
            cls_score = [
                torch.index_select(tcls, 0, no_ignore_idx)
                for tcls in cls_score
            ]

        loss = None

        for i in range(0, len(self.norm_list)):
            if self.norm_list[i]:
                a = torch.nn.functional.normalize(cls_score[0], p=2, dim=1)
            else:
                a = cls_score[0]

            if self.miner_list[i] is not None:
                tuple_indice = self.miner_list[i](a, tlabel)
                if not torch.isnan(self.loss_list[i](a, tlabel, tuple_indice)):
                    if loss is None:
                        loss = self.loss_weight_list[i] * self.loss_list[i](
                            a, tlabel, tuple_indice)
                    else:
                        loss += self.loss_weight_list[i] * self.loss_list[i](
                            a, tlabel, tuple_indice)
                else:
                    logger.info(
                        'MP metric head catch NAN loss in %dth loss !' % i)
            else:
                if not torch.isnan(self.loss_list[i](a, tlabel)):
                    if loss is None:
                        loss = self.loss_weight_list[i] * self.loss_list[i](
                            a, tlabel)
                    else:
                        loss += self.loss_weight_list[i] * self.loss_list[i](
                            a, tlabel)
                else:
                    logger.info(
                        'MP metric head catch NAN loss in %dth loss !' % i)

        if loss is None:
            loss = torch.tensor(
                0.0, requires_grad=True).to(cls_score[0].device)

        losses['loss'] = loss
        try:
            losses['acc'] = accuracy(a, tlabel)
        except:
            pass

        return losses
