#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Optional


def parameter_list(named_parameters,
                   weight_decay: Optional[float] = 0.0,
                   no_decay_bn_filter_bias: Optional[bool] = False,
                   *args,
                   **kwargs):
    with_decay = []
    without_decay = []
    if isinstance(named_parameters, list):
        for n_parameter in named_parameters:
            for p_name, param in n_parameter():
                if (param.requires_grad and len(param.shape) == 1
                        and no_decay_bn_filter_bias):
                    # biases and normalization layer parameters are of len 1
                    without_decay.append(param)
                elif param.requires_grad:
                    with_decay.append(param)
    else:
        for p_name, param in named_parameters():
            if (param.requires_grad and len(param.shape) == 1
                    and no_decay_bn_filter_bias):
                # biases and normalization layer parameters are of len 1
                without_decay.append(param)
            elif param.requires_grad:
                with_decay.append(param)
    param_list = [{'params': with_decay, 'weight_decay': weight_decay}]
    if len(without_decay) > 0:
        param_list.append({'params': without_decay, 'weight_decay': 0.0})
    return param_list
