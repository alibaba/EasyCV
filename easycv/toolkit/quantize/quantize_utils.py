# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from mmcv.parallel import scatter_kwargs
from mmcv.runner import get_dist_info

quantize_config = {
    'device': 'cpu',
    'backend': 'PyTorch',
}


def calib(model, data_loader):
    for cur_iter, data in enumerate(data_loader):
        input_args, kwargs = scatter_kwargs(None, data, [-1])
        with torch.no_grad():
            kwargs[0]['img'] = kwargs[0]['img'].squeeze(dim=0)
            model(kwargs[0]['img'])
            if cur_iter == 2:
                return
