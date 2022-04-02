# Copyright (c) Alibaba, Inc. and its affiliates.
import functools
import os
import pickle
from collections import OrderedDict
from contextlib import contextmanager

import torch
import torch.distributed as dist
from mmcv.parallel import data_parallel as mm_data_parallel
from mmcv.parallel import distributed as mm_distributed
from mmcv.runner.dist_utils import get_dist_info
from torch import nn
from torch.distributed import ReduceOp


def is_master():
    rank, _ = get_dist_info()
    return rank == 0


def local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


@contextmanager
def dist_zero_exec(rank=local_rank()):
    if rank not in [-1, 0]:
        barrier()
    # execute the context after yield, then return here to continue
    yield
    if rank == 0:
        barrier()


def get_num_gpu_per_node():
    """ get number of gpu per node
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    local_rank_tensor = torch.tensor([local_rank], device='cuda')
    torch.distributed.all_reduce(local_rank_tensor, op=ReduceOp.MAX)
    num_gpus = local_rank_tensor.tolist()[0] + 1

    return num_gpus


def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def is_parallel(model):
    return type(model) in (nn.parallel.DataParallel,
                           nn.parallel.DistributedDataParallel,
                           mm_data_parallel.MMDataParallel,
                           mm_distributed.MMDistributedDataParallel)


# For YOLOX
def obj2tensor(pyobj, device='cuda'):
    """Serialize picklable python object to tensor."""
    storage = torch.ByteStorage.from_buffer(pickle.dumps(pyobj))
    return torch.ByteTensor(storage).to(device=device)


def tensor2obj(tensor):
    """Deserialize tensor to picklable python object."""
    return pickle.loads(tensor.cpu().numpy().tobytes())


@functools.lru_cache()
def _get_global_gloo_group():
    """Return a process group based on gloo backend, containing all the ranks
    The result is cached."""
    if dist.get_backend() == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def all_reduce_dict(py_dict, op='sum', group=None, to_float=True):
    """Apply all reduce function for python dict object.

    The code is modified from https://github.com/Megvii-
    BaseDetection/YOLOX/blob/main/yolox/utils/allreduce_norm.py.

    NOTE: make sure that py_dict in different ranks has the same keys and
    the values should be in the same shape.

    Args:
        py_dict (dict): Dict to be applied all reduce op.
        op (str): Operator, could be 'sum' or 'mean'. Default: 'sum'
        group (:obj:`torch.distributed.group`, optional): Distributed group,
            Default: None.
        to_float (bool): Whether to convert all values of dict to float.
            Default: True.

    Returns:
        OrderedDict: reduced python dict object.
    """
    _, world_size = get_dist_info()
    if world_size == 1:
        return py_dict
    if group is None:
        # TODO: May try not to use gloo in the future
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return py_dict

    # all reduce logic across different devices.
    py_key = list(py_dict.keys())
    py_key_tensor = obj2tensor(py_key)
    dist.broadcast(py_key_tensor, src=0)
    py_key = tensor2obj(py_key_tensor)

    tensor_shapes = [py_dict[k].shape for k in py_key]
    tensor_numels = [py_dict[k].numel() for k in py_key]

    if to_float:
        flatten_tensor = torch.cat(
            [py_dict[k].flatten().float() for k in py_key])
    else:
        flatten_tensor = torch.cat([py_dict[k].flatten() for k in py_key])

    dist.all_reduce(flatten_tensor, op=dist.ReduceOp.SUM)
    if op == 'mean':
        flatten_tensor /= world_size

    split_tensors = [
        x.reshape(shape) for x, shape in zip(
            torch.split(flatten_tensor, tensor_numels), tensor_shapes)
    ]
    return OrderedDict({k: v for k, v in zip(py_key, split_tensors)})
