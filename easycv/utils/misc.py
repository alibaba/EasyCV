# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from functools import partial

import mmcv
import numpy as np
from six.moves import map, zip

from easycv.models.backbones.repvgg_yolox_backbone import RepVGGBlock


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


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.
    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.
    Args:
        func (Function): A function that will be applied to a list of
            arguments
    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


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
