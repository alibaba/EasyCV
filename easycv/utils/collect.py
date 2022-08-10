# Copyright (c) Alibaba, Inc. and its affiliates.
import logging

import mmcv
import numpy as np
import torch

from .gather import gather_tensors_batch


def nondist_forward_collect(func, data_loader, length):
    '''Forward and collect network outputs.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        length (int): Expected length of output arrays.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    '''
    results = []
    prog_bar = mmcv.ProgressBar(len(data_loader))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)
        results.append(result)
        prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        if isinstance(results[0][k], torch.Tensor):
            results_all[k] = np.concatenate(
                [batch[k].numpy() for batch in results], axis=0)
        elif isinstance(results[0][k], (np.ndarray, list)):
            results_all[k] = np.concatenate([batch[k] for batch in results],
                                            axis=0)
        else:
            raise ValueError(
                'value of batch prediction dict should only be tensor or list or ndarray, '
                'but get %s is %s.' % (k, type(results[0][k])))

        assert results_all[k].shape[0] == length
    return results_all


def dist_forward_collect(func, data_loader, rank, length, ret_rank=-1):
    '''Forward and collect network outputs in a distributed manner.

    This function performs forward propagation and collects outputs.
    It can be used to collect results, features, losses, etc.

    Args:
        func (function): The function to process data. The output must be
            a dictionary of CPU tensors.
        rank (int): This process id.
        length (int): Expected length of output arrays.
        ret_rank (int): The process that returns.
            Other processes will return None.

    Returns:
        results_all (dict(np.ndarray)): The concatenated outputs.
    '''
    results = []

    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(data_loader))
    for idx, data in enumerate(data_loader):
        with torch.no_grad():
            result = func(**data)  # dict{key: tensor}
        results.append(result)
        if rank == 0:
            prog_bar.update()

    results_all = {}
    for k in results[0].keys():
        if isinstance(results[0][k], torch.Tensor):
            results_cat = np.concatenate(
                [batch[k].cpu().numpy() for batch in results], axis=0)
        elif isinstance(results[0][k], (np.ndarray, list)):
            results_cat = np.concatenate([batch[k] for batch in results],
                                         axis=0)
        else:
            raise ValueError(
                'value of batch prediction dict should only be tensor or list or ndarray, '
                'but get %s is %s.' % (k, type(results[0][k])))

        if ret_rank == -1:
            results_gathered = gather_tensors_batch(results_cat, part_size=20)
            results_strip = np.concatenate(results_gathered, axis=0)[:length]
        else:
            results_gathered = gather_tensors_batch(
                results_cat, part_size=20, ret_rank=ret_rank)
            if rank == ret_rank:
                results_strip = np.concatenate(
                    results_gathered, axis=0)[:length]
            else:
                results_strip = None
        results_all[k] = results_strip

    return results_all
