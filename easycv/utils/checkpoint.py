# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
import os

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import load_checkpoint as mmcv_load_checkpoint
from mmcv.runner.checkpoint import get_state_dict, weights_to_cpu
from torch.optim import Optimizer

from easycv.file import io
from easycv.file.utils import is_url_path
from easycv.framework.errors import TypeError
from easycv.utils.constant import CACHE_DIR


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

    return mmcv_load_checkpoint(
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
