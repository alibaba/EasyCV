# Copyright (c) Alibaba, Inc. and its affiliates.
import torch.nn as nn
from torch.utils.data import Dataset

from easycv.framework.errors import TypeError
from easycv.utils.collect import dist_forward_collect, nondist_forward_collect


class Extractor(object):

    def __init__(self,
                 dataset,
                 imgs_per_gpu,
                 workers_per_gpu,
                 dist_mode=False):
        from easycv import datasets
        if isinstance(dataset, Dataset):
            self.dataset = dataset
        elif isinstance(dataset, dict):
            self.dataset = datasets.build_dataset(dataset)
        else:
            raise TypeError(
                'dataset must be a Dataset object or a dict, not {}'.format(
                    type(dataset)))
        self.data_loader = datasets.build_dataloader(
            self.dataset,
            imgs_per_gpu,
            workers_per_gpu,
            dist=dist_mode,
            shuffle=False)
        self.dist_mode = dist_mode
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def _forward_func(self, runner, **x):
        backbone_feat = runner.model(mode='extract', **x)
        last_layer_feat = runner.model.module.neck([backbone_feat[-1]])[0]
        last_layer_feat = last_layer_feat.view(last_layer_feat.size(0), -1)
        return dict(feature=last_layer_feat.cpu())

    def __call__(self, runner):
        func = lambda **x: self._forward_func(runner, **x)
        if self.dist_mode:
            feats = dist_forward_collect(
                func,
                self.data_loader,
                runner.rank,
                len(self.dataset),
                ret_rank=-1)['feature']  # NxD
        else:
            feats = nondist_forward_collect(func, self.data_loader,
                                            len(self.dataset))['feature']
        return feats
