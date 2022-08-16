# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import get_dist_info

from easycv.utils.preprocess_function import mixUp
from .. import builder
from ..registry import MODELS
from .moco import MOCO


@MODELS.register_module
class MIXCO(MOCO):
    '''MOCO.

        A mixup version moco https://arxiv.org/pdf/2010.06300.pdf
    '''

    def __init__(self,
                 backbone,
                 train_preprocess=[],
                 neck=None,
                 head=None,
                 mixco_head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 **kwargs):

        if 'mixUp' in train_preprocess:
            rank, _ = get_dist_info()
            np.random.seed(rank + 12)
            self.mixup = mixUp
            self.lam = None
            train_preprocess.remove('mixUp')

        super(MIXCO, self).__init__(backbone, train_preprocess, neck, head,
                                    pretrained, queue_len, feat_dim, momentum,
                                    **kwargs)

        self.mixco_head = builder.build_head(mixco_head)

    def forward_train(self, img, **kwargs):
        assert isinstance(img, list)
        assert len(img) == 2
        for _img in img:
            assert _img.dim() == 4, \
                'Input must have 4 dims, got: {}'.format(_img.dim())

        im_q = img[0].contiguous()
        im_k = img[1].contiguous()

        for preprocess in self.train_preprocess:
            im_q = preprocess(im_q)
            im_k = preprocess(im_k)

        if self.mixup is not None:
            im_m, self.lam = self.mixup(im_q)
            mixup_feature = self.encoder_q(im_m)[0]  # queries: N/2 x C
            mixup_feature = nn.functional.normalize(mixup_feature, dim=1)

        # compute query features
        q = self.encoder_q(im_q)[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)[0]  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        losses = self.head(l_pos, l_neg)

        # mixco loss : mixup mocov2 loss
        if self.mixup is not None and self.mixco_head is not None:
            k1 = k[0:mixup_feature.size(0), ...]
            k2 = k[mixup_feature.size(0):, ...]
            # print(k1.size(), k2.size(), self.lam)
            l_pos1 = torch.einsum('nc,nc->n',
                                  [mixup_feature, k1]).unsqueeze(-1)
            l_pos2 = torch.einsum('nc,nc->n',
                                  [mixup_feature, k2]).unsqueeze(-1)

            l_neg1 = torch.einsum(
                'nc,ck->nk',
                [mixup_feature, self.queue.clone().detach()])

            losses['loss'] += self.mixco_head(l_pos1,
                                              l_neg1)['loss'] * self.lam
            losses['loss'] += self.mixco_head(l_pos2,
                                              l_neg1)['loss'] * (1 - self.lam)

        self._dequeue_and_enqueue(k)

        return losses
