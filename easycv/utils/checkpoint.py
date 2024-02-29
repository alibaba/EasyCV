# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os
import re
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from mmcv.parallel import is_module_wrapper
from mmcv.runner import _load_checkpoint as _load_checkpoint
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from torch import distributed as dist
from torch.optim import Optimizer

from easycv.file import io
from easycv.file.utils import is_url_path
from easycv.framework.errors import TypeError
from easycv.utils.constant import CACHE_DIR


def get_checkpoint(filename):
    if filename.startswith('oss://'):
        _, fname = os.path.split(filename)
        cache_file = os.path.join(CACHE_DIR, fname)
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        if not os.path.exists(cache_file):
            logging.info(
                f'download checkpoint from {filename} to {cache_file}')
            io.copy(filename, cache_file)
        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            torch.distributed.barrier()
        filename = cache_file
    elif is_url_path(filename):
        from torch.hub import urlparse, download_url_to_file
        parts = urlparse(filename)
        base_name = os.path.basename(parts.path)
        cache_file = os.path.join(CACHE_DIR, base_name)
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
        if not os.path.exists(cache_file):
            logging.info(
                f'download checkpoint from {filename} to {cache_file}')
            download_url_to_file(filename, cache_file)
        if torch.distributed.is_available(
        ) and torch.distributed.is_initialized():
            torch.distributed.barrier()
        filename = cache_file
    return filename


def load_and_check_state_dict(module: nn.Module,
                              state_dict: Union[dict, OrderedDict],
                              strict: bool = False,
                              logger: Optional[logging.Logger] = None) -> None:
    """Load state_dict to a module.

    This method is modified from :meth:`mmcv.runner.checkpoint.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.
    Raise error when state_dict is highly mismatched.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (dict or OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys: List[str] = []
    all_missing_keys: List[str] = []
    err_msg: List[str] = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()  # type: ignore
    if metadata is not None:
        state_dict._metadata = metadata  # type: ignore

    # use _load_from_state_dict to enable checkpoint version control
    def load(module, prefix=''):
        # recursively check parallel module in case that the model has a
        # complicated structure, e.g., nn.Module(nn.Module(DDP))
        if is_module_wrapper(module):
            module = module.module
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    def get_dist_info() -> Tuple[int, int]:
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        return rank, world_size

    load(module)
    # break load->load reference cycle
    load = None  # type: ignore

    # ignore "num_batches_tracked" of BN layers
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    rank, _ = get_dist_info()
    if len(err_msg) > 0 and rank == 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)  # type: ignore
        if strict:
            raise RuntimeError(err_msg)
        else:
            if logger is not None:
                logger.warning(err_msg)
            else:
                print(err_msg)

            err_msg_list = err_msg.split('\n')

            for error_msg_info in err_msg_list:
                if 'size mismatch' in error_msg_info and 'cls' not in error_msg_info:
                    raise RuntimeError(
                        'Please check your pretrained model. The parameters do not match outside of the cls layer.'
                    )


def load_and_check_checkpoint(
        model: torch.nn.Module,
        filename: str,
        map_location: Union[str, Callable, None] = None,
        strict: bool = False,
        logger: Optional[logging.Logger] = None,
        revise_keys: list = [(r'^module\.', '')]) -> Union[dict, OrderedDict]:
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    load_and_check_state_dict(model, state_dict, strict, logger)
    return checkpoint


def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None,
                    revise_keys=[(r'^module\.', '')]):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.
        revise_keys (list): A list of customized keywords to modify the
            state_dict in checkpoint. Each item is a (pattern, replacement)
            pair of the regular expression operations. Default: strip
            the prefix 'module.' by [(r'^module\\.', '')].

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    filename = get_checkpoint(filename)
    return load_and_check_checkpoint(
        model,
        filename,
        map_location=map_location,
        strict=strict,
        logger=logger,
        revise_keys=revise_keys)


def save_checkpoint(model, filename, optimizer=None, meta=None):
    """Save checkpoint to file.

    The checkpoint will have 3 fields: ``meta``, ``state_dict`` and
    ``optimizer``. By default ``meta`` will contain version and time info.

    Args:
        model (Module): Module whose params are to be saved.
        filename (str): Checkpoint filename.
        optimizer (:obj:`Optimizer`, optional): Optimizer to be saved.
        meta (dict, optional): Metadata to be saved in checkpoint.
    """
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')

    out_dir = os.path.dirname(filename)
    out_dir = out_dir + '/' if out_dir[-1] != '/' else out_dir
    if not io.isdir(out_dir):
        io.makedirs(out_dir)

    if is_module_wrapper(model):
        model = model.module

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model))
    }

    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    with io.open(filename, 'wb') as ofile:
        torch.save(checkpoint, ofile)
