# Copyright (c) Alibaba, Inc. and its affiliates.
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import get_dist_info

from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger
from easycv.utils.preprocess_function import gaussianBlur, randomGrayScale
from .. import builder
from ..base import BaseModel
from ..registry import MODELS


@MODELS.register_module
class SWAV(BaseModel):

    def __init__(self,
                 backbone,
                 train_preprocess=[],
                 neck=None,
                 config=None,
                 pretrained=None):
        super(SWAV, self).__init__()
        self.pretrained = pretrained
        self.backbone = builder.build_backbone(backbone)

        self.preprocess_key_map = {
            'randomGrayScale': randomGrayScale,
            'gaussianBlur': gaussianBlur
        }
        self.train_preprocess = [
            self.preprocess_key_map[i] for i in train_preprocess
        ]
        self.neck = builder.build_neck(neck)
        self.config = config

        self.prototypes = None
        nmb_prototypes = self.config['nmb_prototypes']
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(neck['out_channels'],
                                              nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(
                neck['out_channels'], nmb_prototypes, bias=False)

        self.feat_dim = neck['out_channels']

        self.l2norm = True
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Softmax(dim=1).cuda()
        self.use_the_queue = False
        self.init_weights()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self.backbone, self.pretrained, strict=False, logger=logger)
        else:
            self.backbone.init_weights()
        self.neck.init_weights(init_linear='kaiming')

        # if torch.load(pretrained).get("prototypes.weight", None) is not None:
        #     self.prototypes.weight.data = torch.load(pretrained)['state_dict'].get("prototypes.weight")

    def forward_backbone(self, img):
        feature_list = self.backbone(img)
        return feature_list

    def forward_train_model(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(
            torch.unique_consecutive(
                torch.tensor([inp.shape[-1] for inp in inputs]),
                return_counts=True,
            )[1], 0
        )  # this is a split operation, get the different shape input index

        start_idx = 0
        for end_idx in idx_crops:
            img = torch.cat(inputs[start_idx:end_idx]).cuda(non_blocking=True)
            for preprocess in self.train_preprocess:
                img = preprocess(img)

            _out = self.forward_backbone(img)[
                0]  # resnet return [[n,c,h,w],[],]
            _out = self.avgpool(_out)
            _out = torch.flatten(_out, 1)

            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx

        output = self.neck([output])[0]

        if self.l2norm:
            output = nn.functional.normalize(output, dim=1, p=2)

        if self.prototypes is not None:
            return output, self.prototypes(output)
        return output

    def forward_train(self, inputs):
        self.backbone.train()
        self.neck.train()
        self.prototypes.train()
        # normalize the prototypes
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.prototypes.weight.copy_(w)
        embedding, output = self.forward_train_model(inputs)
        embedding = embedding.detach()
        bs = inputs[0].size(0)
        # swav loss
        loss = 0
        for i, crop_id in enumerate(self.config['crops_for_assign']):
            with torch.no_grad():
                out = output[bs * crop_id:bs * (crop_id + 1)]

                # time to use the queue
                if getattr(self, 'queue', None) is not None:
                    if self.use_the_queue or not torch.all(
                            self.queue[i, -1, :] == 0):
                        self.use_the_queue = True
                        out = torch.cat(
                            (torch.mm(self.queue[i],
                                      self.prototypes.weight.t()), out))
                    # fill the queue
                    self.queue[i, bs:] = self.queue[i, :-bs].clone()
                    self.queue[i, :bs] = embedding[crop_id * bs:(crop_id + 1) *
                                                   bs]
                # get assignments
                q = torch.exp(out / self.config['epsilon']).t()
                q = distributed_sinkhorn(
                    q, self.config['sinkhorn_iterations'])[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(
                    np.arange(np.sum(self.config['num_crops'])), crop_id):
                p = self.softmax(output[bs * v:bs * (v + 1)] /
                                 self.config['temperature'])
                subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
            loss += subloss / (np.sum(self.config['num_crops']) - 1)
        loss /= len(self.config['crops_for_assign'])
        losses = dict()
        losses['loss'] = loss
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


class MultiPrototypes(nn.Module):

    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module('prototypes' + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


def distributed_sinkhorn(Q, nmb_iters):
    rank, world_size = get_dist_info()
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        dist.all_reduce(sum_Q)
        Q /= sum_Q

        # u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(non_blocking=True) / (
            world_size * Q.shape[1])

        curr_sum = torch.sum(Q, dim=1)
        dist.all_reduce(curr_sum)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()
