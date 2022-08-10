# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class ContrastiveHead(nn.Module):
    '''Head for contrastive learning.
    '''

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        '''
        Args:
            pos (Tensor): Nx1 positive similarity
            neg (Tensor): Nxk negative similarity
        '''
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        losses = dict()
        losses['loss'] = self.criterion(logits, labels)
        return losses


@HEADS.register_module
class DebiasedContrastiveHead(nn.Module):

    def __init__(self, temperature=0.1, tau=0.1):
        super(DebiasedContrastiveHead, self).__init__()
        self.temperature = temperature
        self.tau = tau

    def forward(self, pos, neg):
        '''
        Args:
            pos (Tensor): Nx1 positive similarity
            neg (Tensor): Nxk negative similarity
        '''
        bs = pos.size(0)
        N = bs * 2 - 2
        Ng = (-self.tau * N * pos + neg.sum(dim=-1)) / (1 - self.tau)
        Ng = torch.clamp(Ng, min=N * np.e**(-1 / self.temperature))

        loss = (-torch.log(pos / (pos + Ng))).mean()

        losses = dict()
        losses['loss'] = loss
        return losses
