# Copyright (c) Alibaba, Inc. and its affiliates.
import platform
import random
from distutils.version import LooseVersion
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader

from easycv.datasets.builder import build_sampler
from easycv.datasets.shared.odps_reader import set_dataloader_workid
from easycv.framework.errors import NotImplementedError
from easycv.utils.dist_utils import sync_random_seed
from easycv.utils.torchacc_util import is_torchacc_enabled
from .collate import CollateWrapper

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     imgs_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     replace=False,
                     seed=None,
                     reuse_worker_cache=False,
                     odps_config=None,
                     persistent_workers=False,
                     collate_hooks=None,
                     use_repeated_augment_sampler=False,
                     sampler=None,
                     pin_memory=False,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        imgs_per_gpu (int): Number of images on each GPU, i.e., batch size of
            each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        replace (bool): Replace or not in random shuffle.
            It works on when shuffle is True.
        seed (int, Optional): The seed. Default to None.
        reuse_worker_cache (bool): If set true, will reuse worker process so that cached
            data in worker process can be reused.
        persistent_workers (bool) : After pytorch1.7, could use persistent_workers=True to
            avoid reconstruct dataworker before each epoch, speed up before epoch
        use_repeated_augment_sampler (bool) : If set true, it will use RASampler.
            Default: False.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()

    if dist:
        seed = sync_random_seed(seed)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if replace:
            raise NotImplementedError
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    default_sampler_args = dict(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=seed,
        replace=replace)

    split_huge_listfile_byrank = getattr(dataset, 'split_huge_listfile_byrank',
                                         False)

    if sampler is not None:
        sampler_cfg = sampler
        sampler_cfg.update(default_sampler_args)
    elif use_repeated_augment_sampler:
        sampler_cfg = dict(type='RASampler', **default_sampler_args)
    elif hasattr(dataset, 'm_per_class') and dataset.m_per_class > 1:
        sampler_cfg = dict(
            type='DistributedMPSampler',
            split_huge_listfile_byrank=split_huge_listfile_byrank,
            **default_sampler_args)
    else:
        if dist:
            sampler_cfg = dict(
                type='DistributedSampler',
                split_huge_listfile_byrank=split_huge_listfile_byrank,
                **default_sampler_args)
        else:
            sampler_cfg = dict(
                type='RandomSampler',
                data_source=dataset) if shuffle else None  # TODO: set replace

    sampler = build_sampler(sampler_cfg) if sampler_cfg is not None else None

    init_fn = partial(
        worker_init_fn,
        num_workers=num_workers,
        rank=rank,
        seed=seed,
        odps_config=odps_config) if seed is not None else None
    collate_fn = dataset.collate_fn if hasattr(
        dataset, 'collate_fn') else partial(
            collate, samples_per_gpu=imgs_per_gpu)

    if collate_hooks:
        collate_fn = CollateWrapper(collate_fn, collate_hooks)

    if not reuse_worker_cache:
        if LooseVersion(torch.__version__) < LooseVersion('1.7.0'):
            print(
                'Pytorch Version < 1.7, build Dataloader without persistent_workers'
            )
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                worker_init_fn=init_fn,
                **kwargs)
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=pin_memory,
                worker_init_fn=init_fn,
                persistent_workers=persistent_workers,
                **kwargs)
    else:
        # use InfiniteDataLoader to reuse worker process for caching data
        data_loader = InfiniteDataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            worker_init_fn=init_fn,
            **kwargs)

    if is_torchacc_enabled():
        from .loader_wrapper import TorchaccLoaderWrapper
        data_loader = TorchaccLoaderWrapper(data_loader)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed, odps_config=None):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    if odps_config is not None:
        # for odps to set correct offset in multi-process pytorch dataloader
        # use init_fn to set global DATALOADER_WORKID before dataset getitem
        set_dataloader_workid(worker_id)
        # set_dataloader_worknum(imgs_per_gpu)


class InfiniteDataLoader(torch.utils.data.dataloader.DataLoader):
    """ Dataloader that reuses workers. https://github.com/pytorch/pytorch/issues/15849
    Uses same syntax as vanilla DataLoader.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler',
                           _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)
