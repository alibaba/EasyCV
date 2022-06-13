# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import get_dist_info
from timm.data.mixup import Mixup

from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.logger import get_root_logger, print_log
from easycv.utils.preprocess_function import (bninceptionPre, gaussianBlur,
                                              mixUpCls, randomErasing)
from .. import builder
from ..base import BaseModel
from ..registry import MODELS
from ..utils import Sobel


@MODELS.register_module
class Classification(BaseModel):
    """
    Args:
        pretrained: Select one {str or True or False/None}.
        if pretrained == str, load model from specified path;
        if pretrained == True, load model from default path(currently only supports timm);
        if pretrained == False or None, load from init weights.
    """

    def __init__(self,
                 backbone,
                 train_preprocess=[],
                 with_sobel=False,
                 head=None,
                 neck=None,
                 pretrained=True,
                 mixup_cfg=None):
        super(Classification, self).__init__()
        self.with_sobel = with_sobel
        self.pretrained = pretrained
        if with_sobel:
            self.sobel_layer = Sobel()
        else:
            self.sobel_layer = None

        self.preprocess_key_map = {
            'bninceptionPre': bninceptionPre,
            'gaussianBlur': gaussianBlur,
            'mixUpCls': mixUpCls,
            'randomErasing': randomErasing
        }

        if 'mixUp' in train_preprocess:
            rank, _ = get_dist_info()
            np.random.seed(rank + 12)
            if not mixup_cfg:
                num_classes = head.get(
                    'num_classes',
                    1000) if 'num_classes' in head else backbone.get(
                        'num_classes', 1000)
                mixup_cfg = dict(
                    mixup_alpha=0.8,
                    cutmix_alpha=1.0,
                    cutmix_minmax=None,
                    prob=1.0,
                    switch_prob=0.5,
                    mode='batch',
                    label_smoothing=0.1,
                    num_classes=num_classes)
            self.mixup = Mixup(**mixup_cfg)
            head.loss_config = {'type': 'SoftTargetCrossEntropy'}
            train_preprocess.remove('mixUp')
        self.train_preprocess = [
            self.preprocess_key_map[i] for i in train_preprocess
        ]

        self.backbone = builder.build_backbone(backbone)

        assert head is not None, 'Classification head should be configed'

        if type(head) == list:
            self.head_num = len(head)
            tmp_head_list = [builder.build_head(h) for h in head]
        else:
            self.head_num = 1
            tmp_head_list = [builder.build_head(head)]

        # do this setattr to make sure nn.Module to be attr of nn.Module
        for idx, h in enumerate(tmp_head_list):
            setattr(self, 'head_%d' % idx, h)

        if type(neck) == list:
            self.neck_num = len(neck)
            tmp_neck_list = [builder.build_neck(n) for n in neck]
        elif neck is not None:
            self.neck_num = 1
            tmp_neck_list = [builder.build_neck(neck)]
        else:
            self.neck_num = 0
            tmp_neck_list = []

        # do this setattr to make sure nn.Module to be attr of nn.Module
        for idx, n in enumerate(tmp_neck_list):
            setattr(self, 'neck_%d' % idx, n)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.activate_fn = nn.Softmax(dim=1)
        self.extract_list = ['neck']

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

        for idx in range(self.head_num):
            h = getattr(self, 'head_%d' % idx)
            h.init_weights()

        for idx in range(self.neck_num):
            n = getattr(self, 'neck_%d' % idx)
            n.init_weights()

    def forward_backbone(self, img: torch.Tensor) -> List[torch.Tensor]:
        """Forward backbone

        Returns:
            x (tuple): backbone outputs
        """
        if self.sobel_layer is not None:
            img = self.sobel_layer(img)
        x = self.backbone(img)
        return x

    @torch.jit.unused
    def forward_train(self, img, gt_labels) -> Dict[str, torch.Tensor]:
        """
            In forward train, model will forward backbone + neck / multi-neck, get alist of output tensor,
            and put this list to head / multi-head, to compute each loss
        """
        # for mxk sampler, use datasource type =  ClsSourceImageListByClass will sample k img in 1 class,
        #  input data will be m x k x c x h x w, should be reshape to (m x k) x c x h x w
        if img.dim() == 5:
            new_shape = [
                img.shape[0] * img.shape[1], img.shape[2], img.shape[3],
                img.shape[4]
            ]
            img = img.view(new_shape)
            gt_labels = gt_labels.view([-1])

        for preprocess in self.train_preprocess:
            img = preprocess(img)

        if hasattr(self, 'mixup'):
            img, gt_labels = self.mixup(img, gt_labels)

        x = self.forward_backbone(img)

        if self.neck_num > 0:
            tmp = []
            for idx in range(self.neck_num):
                h = getattr(self, 'neck_%d' % idx)
                tmp += h(x)
            x = tmp
        else:
            x = x

        losses = {}
        for idx in range(self.head_num):
            h = getattr(self, 'head_%d' % idx)
            outs = h(x)
            loss_inputs = (outs, gt_labels)
            hlosses = h.loss(*loss_inputs)
            if 'loss' in losses.keys():
                losses['loss'] += hlosses['loss']
            else:
                losses['loss'] = hlosses['loss']

        return losses

    # @torch.jit.unused
    def forward_test(self, img: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
            forward_test means generate prob/class from image only support one neck  + one head
        """
        x = self.forward_backbone(img)  # tuple

        # if self.neck_num > 0:
        if hasattr(self, 'neck_0'):
            x = self.neck_0([i for i in x])

        out = self.head_0(x)[0].cpu()
        result = {}
        result['prob'] = self.activate_fn(out)
        result['class'] = torch.argmax(result['prob'])
        return result

    @torch.jit.unused
    def forward_test_label(self, img, gt_labels) -> Dict[str, torch.Tensor]:
        """
            forward_test_label means generate prob/class from image only support one neck  + one head
            ps : head init need set the input feature idx
        """
        x = self.forward_backbone(img)  # tuple

        if hasattr(self, 'neck_0'):
            x = self.neck_0([i for i in x])

        out = [self.head_0(x)[0].cpu()]
        keys = ['neck']
        keys.append('gt_labels')
        out.append(gt_labels.cpu())
        return dict(zip(keys, out))

    def aug_test(self, imgs):
        raise NotImplementedError

    def forward_feature(self, img) -> Dict[str, torch.Tensor]:
        """Forward feature  means forward backbone  + neck/multineck ,get dict of output feature,
            self.neck_num = 0: means only forward backbone, output backbone feature with avgpool, with key neck,
            self.neck_num > 0: means has 1/multi neck, output neck's feature with key neck_neckidx_featureidx, suck as neck_0_0
        Returns:
            x (torch.Tensor): feature tensor
        """
        return_dict = {}
        x = self.backbone(img)
        # return_dict['backbone'] = x[-1]
        if hasattr(self, 'neck_0'):
            tmp = []
            for idx in range(self.neck_num):
                neck_name = 'neck_%d' % idx
                h = getattr(self, neck_name)
                neck_h = h([i for i in x])
                tmp = tmp + neck_h
                for j in range(len(neck_h)):
                    neck_name = 'neck_%d_%d' % (idx, j)
                    return_dict['neck_%d_%d' % (idx, j)] = neck_h[j]
                    if neck_name not in self.extract_list:
                        self.extract_list.append(neck_name)
            return_dict['neck'] = tmp[0]
        else:
            feature = self.avg_pool(x[-1])
            feature = feature.view(feature.size(0), -1)
            return_dict['neck'] = feature

        return return_dict

    def update_extract_list(self, key):
        if key not in self.extract_list:
            self.extract_list.append(key)
        return

    def forward(
        self,
        img: torch.Tensor,
        mode: str = 'train',
        gt_labels: Optional[torch.Tensor] = None,
        img_metas: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # TODO: support Dict[any, any] type of img_metas
        del img_metas  # fake img_metas for support jit
        if mode == 'train':
            assert gt_labels is not None
            return self.forward_train(img, gt_labels)
        elif mode == 'test':
            if gt_labels is not None:
                return self.forward_test_label(img, gt_labels)
            else:
                return self.forward_test(img)
        elif mode == 'extract':
            rd = self.forward_feature(img)
            rv = {}
            for name in self.extract_list:
                if name in rd.keys():
                    rv[name] = rd[name].cpu()
                else:
                    raise ValueError(
                        'Extract {} is not support in classification models'.
                        format(name))
            if gt_labels is not None:
                rv['gt_labels'] = gt_labels.cpu()
            return rv
        else:
            raise Exception('No such mode: {}'.format(mode))
