# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import inspect
import logging
import pickle
import warnings

import mmcv
import numpy as np
import torch

from easycv.framework.errors import ValueError


def tensor2imgs(tensor, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True):
    num_imgs = tensor.size(0)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    for img_id in range(num_imgs):
        img = tensor[img_id, ...].cpu().numpy().transpose(1, 2, 0)
        img = mmcv.imdenormalize(
            img, mean, std, to_bgr=to_rgb).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    return imgs


def unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new_full((count, ), fill)
        ret[inds] = data
    else:
        new_size = (count, ) + data.size()[1:]
        ret = data.new_full(new_size, fill)
        ret[inds, :] = data
    return ret


def add_prefix(inputs, prefix):
    """Add prefix for dict key.

    Args:
        inputs (dict): The input dict with str keys.
        prefix (str): The prefix add to key name.

    Returns:
        dict: The dict with keys wrapped with ``prefix``.
    """

    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value

    return outputs


def reparameterize_models(model):
    """ reparameterize model for inference, especially forf
            1. rep conv block : merge 3x3 weight 1x1 weights
        call module switch_to_deploy recursively
    Args:
        model: nn.Module
    """
    from easycv.models.backbones.repvgg_yolox_backbone import RepVGGBlock

    reparameterize_count = 0
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            reparameterize_count += 1
            layer.switch_to_deploy()
    logging.info(
        'export : PAI-export reparameterize_count(RepVGGBlock, ) switch to deploy with {} blocks'
        .format(reparameterize_count))
    print('reparam:', reparameterize_count)
    return model


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    def decorator(func1):
        if inspect.isclass(func1):
            fmt1 = 'Call to deprecated class {name} ({reason}).'
        else:
            fmt1 = 'Call to deprecated function {name} ({reason}).'

        @functools.wraps(func1)
        def new_func1(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt1.format(name=func1.__name__, reason=reason),
                category=DeprecationWarning,
                stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)
            return func1(*args, **kwargs)

        return new_func1

    return decorator


def encode_str_to_tensor(obj):
    if isinstance(obj, str):
        return torch.tensor(bytearray(pickle.dumps(obj)), dtype=torch.uint8)
    elif isinstance(obj, torch.Tensor):
        return obj
    else:
        raise ValueError(f'Not support type {type(obj)}')


def decode_tensor_to_str(obj):
    if isinstance(obj, torch.Tensor):
        return pickle.loads(obj.cpu().numpy().tobytes())
    elif isinstance(obj, str):
        return obj
    else:
        raise ValueError(f'Not support type {type(obj)}')
