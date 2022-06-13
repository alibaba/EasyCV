# Copyright (c) Alibaba, Inc. and its affiliates.
import platform
import random
from distutils.version import LooseVersion
from functools import partial

import numpy as np
import torch
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader, RandomSampler

from easycv.datasets.shared.odps_reader import set_dataloader_workid
from .sampler import DistributedMPSampler, DistributedSampler

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
        reuse_worker_cache (bool): If set true, will reuse worker process so that cached
            data in worker process can be reused.
        persistent_workers (bool) : After pytorch1.7, could use persistent_workers=True to
            avoid reconstruct dataworker before each epoch, speed up before epoch
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """

    if dist:
        rank, world_size = get_dist_info()
        split_huge_listfile_byrank = getattr(dataset,
                                             'split_huge_listfile_byrank',
                                             False)

        if hasattr(dataset, 'm_per_class') and dataset.m_per_class > 1:
            sampler = DistributedMPSampler(
                dataset,
                world_size,
                rank,
                shuffle=shuffle,
                split_huge_listfile_byrank=split_huge_listfile_byrank)
        else:
            sampler = DistributedSampler(
                dataset,
                world_size,
                rank,
                shuffle=shuffle,
                split_huge_listfile_byrank=split_huge_listfile_byrank)
        batch_size = imgs_per_gpu
        num_workers = workers_per_gpu
    else:
        if replace:
            raise NotImplementedError
        if hasattr(dataset, 'm_per_class') and dataset.m_per_class > 1:
            sampler = DistributedMPSampler(
                dataset, 1, 0, shuffle=shuffle, replace=replace)
        else:
            sampler = RandomSampler(
                dataset) if shuffle else None  # TODO: set replace
        batch_size = num_gpus * imgs_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(worker_init_fn, seed=seed, odps_config=odps_config)
    collate_fn = dataset.collate_fn if hasattr(
        dataset, 'collate_fn') else partial(
            collate, samples_per_gpu=imgs_per_gpu)

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
                pin_memory=False,
                worker_init_fn=init_fn,
                **kwargs)
        else:
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=False,
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
            pin_memory=False,
            worker_init_fn=init_fn,
            **kwargs)

    return data_loader


def worker_init_fn(worker_id, seed=None, odps_config=None):
    if seed is not None:
        worker_seed = worker_id + seed
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
