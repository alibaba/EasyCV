# Copyright (c) Alibaba, Inc. and its affiliates.
from torch import optim


def build_yolo_optimizer(model, optimizer_cfg):
    """ build optimizer for yolo.
    """
    if hasattr(model, 'module'):
        model = model.module

    pg0, pg1, pg2 = [], [], []
    optimizer_cfg = optimizer_cfg.copy()
    for k, v in model.named_parameters():
        v.requires_grad = True
        if '.bias' in k:
            pg2.append(v)  # biases
        elif '.weight' in k and '.bn' not in k:
            pg1.append(v)  # apply weight decay
        else:
            pg0.append(v)  # all else

    if optimizer_cfg.type == 'Adam':
        optimizer = optim.Adam(
            pg0, lr=optimizer_cfg.lr,
            betas=(optimizer_cfg.momentum, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(
            pg0,
            lr=optimizer_cfg.lr,
            momentum=optimizer_cfg.momentum,
            nesterov=optimizer_cfg.nesterov)

    optimizer.add_param_group({
        'params': pg1,
        'weight_decay': optimizer_cfg.weight_decay
    })  # add pg1 with weight_decay
    # add pg2 (biases), biases are kept in one group to set a special learning rate policy
    # see YoloLrUpdaterHook in easycv/hooks/lr_hook.py
    optimizer.add_param_group({'params': pg2})
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' %
          (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    return optimizer
