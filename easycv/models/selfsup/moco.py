# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn

from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger
from easycv.utils.preprocess_function import gaussianBlur, randomGrayScale
from .. import builder
from ..base import BaseModel
from ..registry import MODELS


@MODELS.register_module
class MOCO(BaseModel):
    '''MOCO.
        Part of the code is borrowed from:
        https://github.com/facebookresearch/moco/blob/master/moco/builder.py.
    '''

    def __init__(self,
                 backbone,
                 train_preprocess=[],
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 **kwargs):
        super(MOCO, self).__init__()

        self.pretrained = pretrained

        self.preprocess_key_map = {
            'randomGrayScale': randomGrayScale,
            'gaussianBlur': gaussianBlur
        }
        self.train_preprocess = [
            self.preprocess_key_map[i] for i in train_preprocess
        ]
        self.encoder_q = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.encoder_k = nn.Sequential(
            builder.build_backbone(backbone), builder.build_neck(neck))
        self.backbone = self.encoder_q[0]
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        self.head = builder.build_head(head)
        self.init_weights()

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self.encoder_q[0],
                self.pretrained,
                strict=False,
                logger=logger)
        else:
            self.encoder_q[0].init_weights()
        self.encoder_q[1].init_weights(init_linear='kaiming')
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)

    def forward_backbone(self, img):
        feature_list = self.backbone(img)
        return feature_list

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

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
        self._dequeue_and_enqueue(k)

        return losses

    def forward_test(self, img, **kwargs):
        pass

    def forward_feature(self, img, **kwargs):
        """Forward backbone

        Returns:
            x (torch.Tensor): feature tensor
        """
        return_dict = {}
        x = self.backbone(img)
        return_dict['backbone'] = x

        if hasattr(self, 'neck') and self.neck is not None:
            feature = self.neck([self.avg_pool(i) for i in x])[0]
        else:
            feature = self.avg_pool(x[-1])
        return_dict['neck'] = feature

        return return_dict

    def forward(self,
                img,
                gt_label=None,
                mode='train',
                extract_list=['neck'],
                **kwargs):
        if mode == 'train':
            return self.forward_train(img, **kwargs)
        elif mode == 'test':
            return self.forward_test(img, **kwargs)

        elif mode == 'extract':
            rd = self.forward_feature(img)
            rv = {}

            for name in extract_list:
                if name in rd.keys():
                    rv[name] = rd[name]
                else:
                    raise 'Extract %s is not support in classification models' % name
            if gt_label is not None:
                rv['gt_labels'] = gt_label.cpu()
            return rv
        else:
            raise Exception('No such mode: {}'.format(mode))


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
