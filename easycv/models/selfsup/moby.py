# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F

from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger
from easycv.utils.preprocess_function import gaussianBlur, randomGrayScale
from .. import builder
from ..base import BaseModel
from ..registry import MODELS


@MODELS.register_module
class MoBY(BaseModel):
    '''MoBY.
        Part of the code is borrowed from:
        https://github.com/SwinTransformer/Transformer-SSL/blob/main/models/moby.py.
    '''

    def __init__(self,
                 backbone,
                 train_preprocess=[],
                 neck=None,
                 head=None,
                 pretrained=None,
                 queue_len=4096,
                 contrast_temperature=0.2,
                 momentum=0.99,
                 online_drop_path_rate=0.2,
                 target_drop_path_rate=0.0,
                 **kwargs):
        """ Init Moby

        Args:
            backbone: backbone config to build vision backbone
            train_preprocess: [gaussBlur, mixUp, solarize]
            neck: neck config to build Moby Neck
            head: head config to build Moby Neck
            pretrained: pretrained weight for backbone
            queue_len :  moby queue length
            contrast_temperature : contrastive_loss temperature
            momentum : ema target weights momentum
            online_drop_path_rate: for transformer based backbone, set online model drop_path_rate
            target_drop_path_rate: for transformer based backbone, set target model drop_path_rate
        """

        super(MoBY, self).__init__()

        self.pretrained = pretrained

        self.preprocess_key_map = {
            'randomGrayScale': randomGrayScale,
            'gaussianBlur': gaussianBlur
        }
        self.train_preprocess = [
            self.preprocess_key_map[i] for i in train_preprocess
        ]

        # build model
        if backbone.get('drop_path_rate', None) is not None:
            backbone['drop_path_rate'] = online_drop_path_rate
        self.encoder_q = builder.build_backbone(backbone)

        if backbone.get('drop_path_rate', None) is not None:
            backbone['drop_path_rate'] = target_drop_path_rate
        self.encoder_k = builder.build_backbone(backbone)

        self.backbone = self.encoder_q

        self.projector_q = builder.build_neck(neck)
        self.projector_k = builder.build_neck(neck)
        self.predictor = builder.build_neck(head)

        # copy param, set stop_grad
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient
        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # convert bn to sync bn
        try:
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_q)
            nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder_k)
        except Exception as e:
            print('Convert encode BN to syncBN failed for MoBY backbone %s' %
                  (str(type(self.encoder_q))))

        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_q)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.projector_k)
        nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

        # set parameters
        self.init_weights()
        self.queue_len = queue_len
        self.momentum = momentum
        self.contrast_temperature = contrast_temperature
        self.feat_dim = head.get('out_channels', 256)
        assert neck.get('out_channels', 256) == head.get(
            'out_channels',
            256), 'MoBY head and neck should set same output dim'

        # create the queue
        self.register_buffer('queue1',
                             torch.randn(self.feat_dim, self.queue_len))
        self.register_buffer('queue2',
                             torch.randn(self.feat_dim, self.queue_len))
        self.queue1 = F.normalize(self.queue1, dim=0)
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self.encoder_q, self.pretrained, strict=False, logger=logger)
        else:
            self.encoder_q.init_weights()
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
        # need a scheduler
        _contrast_momentum = self.momentum
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                1. - _contrast_momentum)

        for param_q, param_k in zip(self.projector_q.parameters(),
                                    self.projector_k.parameters()):
            param_k.data = param_k.data * _contrast_momentum + param_q.data * (
                1. - _contrast_momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        keys1 = concat_all_gather(keys1)
        keys2 = concat_all_gather(keys2)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[:, ptr:ptr + batch_size] = keys1.transpose(0, 1)
        self.queue2[:, ptr:ptr + batch_size] = keys2.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer
        self.queue_ptr[0] = ptr

    def contrastive_loss(self, q, k, queue):

        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.contrast_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return F.cross_entropy(logits, labels)

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
        feat_1 = self.encoder_q(im_q)  # queries: NxC
        proj_1 = self.projector_q(feat_1)
        pred_1 = self.predictor(proj_1)[0]
        pred_1 = F.normalize(pred_1, dim=1)

        feat_2 = self.encoder_q(im_k)
        proj_2 = self.projector_q(feat_2)
        pred_2 = self.predictor(proj_2)[0]
        pred_2 = F.normalize(pred_2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            feat_1_ng = self.encoder_k(im_q)  # keys: NxC
            proj_1_ng = self.projector_k(feat_1_ng)[0]
            proj_1_ng = F.normalize(proj_1_ng, dim=1)

            feat_2_ng = self.encoder_k(im_k)
            proj_2_ng = self.projector_k(feat_2_ng)[0]
            proj_2_ng = F.normalize(proj_2_ng, dim=1)

        # compute loss
        losses = dict()

        losses['loss'] = self.contrastive_loss(pred_1, proj_2_ng, self.queue2) \
            + self.contrastive_loss(pred_2, proj_1_ng, self.queue1)

        self._dequeue_and_enqueue(proj_1_ng, proj_2_ng)

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
