# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import division
import math
import random

import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from torch.utils.data import DistributedSampler as _DistributedSampler
from torch.utils.data import RandomSampler, Sampler

from easycv.datasets.registry import SAMPLERS
from easycv.framework.errors import ValueError
from easycv.utils.dist_utils import local_rank

SAMPLERS.register_module(RandomSampler)


@SAMPLERS.register_module()
class DistributedMPSampler(_DistributedSampler):

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 split_huge_listfile_byrank=False,
                 **kwargs):
        """ A Distribute sampler which support sample m instance from one class once for classification dataset
            dataset: pytorch dataset object
            num_replicas (optional): Number of processes participating in
                distributed training.
            rank (optional): Rank of the current process within num_replicas.
            shuffle (optional): If true (default), sampler will shuffle the indices
            split_huge_listfile_byrank: if split, return all indice for each rank, because list for each rank has been
                split before build dataset in dist training
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)

        self.local_rank = local_rank()
        self.shuffle = shuffle
        self.unif_sampling_flag = False
        self.split_huge_listfile_byrank = split_huge_listfile_byrank
        self.get_label_dict()

    def __iter__(self):
        # deterministically shuffle based on epoch
        indice_list = self.generate_indice()
        return iter(indice_list)

    def generate_indice(self):
        if self.shuffle:
            random.shuffle(self.label_list)
            for k in self.label_dict.keys():
                random.shuffle(self.label_dict[k])

        this_label_list, this_label_list_size = self.calculate_this_label_list(
        )
        if self.rank == 0:
            print('Each epoch has %d buckets of M imgs for per class' %
                  (self.buckets_num))

        m_per_class = self.dataset.m_per_class
        indice_list = []  # [this_label_list_size x (m * buckets_num)]

        for label in this_label_list:
            idx_list = self.label_dict[label]

            if len(idx_list) < self.buckets_num * m_per_class:
                # this place need(could) add more  random .
                idx_list = idx_list * int(self.buckets_num * m_per_class /
                                          len(idx_list) + 1)

            idx_list = idx_list[0:self.buckets_num * m_per_class]
            indice_list.append(idx_list)

        indice_list = np.array(indice_list).reshape(
            (this_label_list_size * self.buckets_num), m_per_class)
        if self.shuffle:
            np.random.shuffle(indice_list)

        indice_list = list(indice_list.astype(int).flatten())

        return indice_list

    def get_label_dict(self):
        self.label_dict = {}
        self.label_list = []

        if not self.dataset.data_source.has_labels:
            raise ValueError(
                'MPSampler need initial with classification datasets which has label!'
            )

        for idx, label in enumerate(self.dataset.data_source.labels):
            if label in self.label_dict.keys():
                self.label_dict[label].append(idx)
            else:
                self.label_dict[label] = [idx]
                self.label_list.append(label)

        if self.rank == 0:
            print(
                self.rank, ' : Total %d Label in %s' %
                (len(self.label_list), type(self.dataset)))

        # calculate the After mpsampler, dataset length change and buckets_num
        self.calculate_this_label_list()
        if self.rank == 0:
            print('Before original dataset length is  %d' %
                  len(self.dataset.data_source))
            print('After  MPRefine dataset length is  %d' % (self.length))
            print('Total %d Label in %s' %
                  (len(self.label_list), type(self.dataset)))

        return

    def calculate_this_label_list(self):
        label_size = len(self.label_list)

        if not self.split_huge_listfile_byrank:
            refine_label_size = int(1 + label_size /
                                    self.num_replicas) * self.num_replicas
            refine_label_list = self.label_list + self.label_list[0:(
                refine_label_size - label_size)]
            this_label_list_size = int(
                len(refine_label_list) / self.num_replicas)
            this_label_list = refine_label_list[self.rank *
                                                this_label_list_size:
                                                (self.rank + 1) *
                                                this_label_list_size]
            m_per_class = self.dataset.m_per_class
            self.buckets_num = int(
                int(len(self.dataset.data_source) / self.num_replicas) /
                (m_per_class * this_label_list_size)) + 1
            self.length = self.buckets_num * m_per_class * int(
                1 +
                len(self.label_list) / self.num_replicas)  # self.num_replicas
        else:
            this_label_list = self.label_list
            this_label_list_size = label_size
            m_per_class = self.dataset.m_per_class

            # this is a huge bug for split situation
            buckets_num = torch.Tensor([
                int(
                    len(self.dataset.data_source) /
                    (m_per_class * this_label_list_size))
            ]).to(self.local_rank)
            torch.distributed.all_reduce(buckets_num,
                                         torch.distributed.ReduceOp.MIN)
            torch.distributed.barrier()
            self.buckets_num = int(max(buckets_num, 1))
            self.length = self.buckets_num * m_per_class * int(
                len(self.label_list))

        return this_label_list, this_label_list_size

    def __len__(self):
        return self.length


@SAMPLERS.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        replace=False,
        split_huge_listfile_byrank=False,
    ):
        """ A Distribute sampler which support sample m instance from one class once for classification dataset
        Args:
            dataset: pytorch dataset object
            num_replicas (optional): Number of processes participating in
                distributed training.
            rank (optional): Rank of the current process within num_replicas.
            shuffle (optional): If true (default), sampler will shuffle the indices
            seed (int, Optional): The seed. Default to 0.
            split_huge_listfile_byrank: if split, return all indice for each rank, because list for each rank has been
                split before build dataset in dist training
        """
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)
        self.shuffle = shuffle
        self.seed = seed
        self.replace = replace
        self.unif_sampling_flag = False
        self.split_huge_listfile_byrank = split_huge_listfile_byrank

    def __iter__(self):
        # deterministically shuffle based on epoch
        if not self.unif_sampling_flag:
            self.generate_new_list()
        else:
            self.unif_sampling_flag = False

        if not self.split_huge_listfile_byrank:
            return iter(
                self.indices[self.rank * self.num_samples:(self.rank + 1) *
                             self.num_samples])
        else:
            return iter(self.indices)

    def generate_new_list(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            if self.replace:
                indices = torch.randint(
                    low=0,
                    high=len(self.dataset),
                    size=(len(self.dataset), ),
                    generator=g).tolist()
            else:
                indices = torch.randperm(
                    len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        self.indices = indices

    def set_uniform_indices(self, labels, num_classes):
        self.unif_sampling_flag = True
        assert self.shuffle, 'Using uniform sampling, the indices must be shuffled.'
        np.random.seed(self.epoch)
        assert (len(labels) == len(self.dataset))
        N = len(labels)
        size_per_label = int(N / num_classes) + 1
        indices = []
        images_lists = [[] for i in range(num_classes)]
        for i, l in enumerate(labels):
            images_lists[l].append(i)
        for i, l in enumerate(images_lists):
            if len(l) == 0:
                continue
            indices.extend(
                np.random.choice(
                    l, size_per_label, replace=(len(l) <= size_per_label)))
        indices = np.array(indices)
        np.random.shuffle(indices)
        indices = indices[:N].astype(np.int).tolist()

        # add extra samples to make it evenly divisible
        assert len(indices) <= self.total_size, \
            '{} vs {}'.format(len(indices), self.total_size)
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size, \
            '{} vs {}'.format(len(indices), self.total_size)
        self.indices = indices

    def __len__(self):
        return self.num_samples if not self.split_huge_listfile_byrank else self.num_samples * self.num_replicas


@SAMPLERS.register_module()
class GroupSampler(Sampler):

    def __init__(self, dataset, samples_per_gpu=1):
        assert hasattr(dataset, 'flag')
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = dataset.flag.astype(np.int64)
        self.group_sizes = np.bincount(self.flag)
        self.num_samples = 0
        for i, size in enumerate(self.group_sizes):
            self.num_samples += int(np.ceil(
                size / self.samples_per_gpu)) * self.samples_per_gpu

    def __iter__(self):
        indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            indice = np.where(self.flag == i)[0]
            assert len(indice) == size
            np.random.shuffle(indice)
            num_extra = int(np.ceil(size / self.samples_per_gpu)
                            ) * self.samples_per_gpu - len(indice)
            indice = np.concatenate(
                [indice, np.random.choice(indice, num_extra)])
            indices.append(indice)
        indices = np.concatenate(indices)
        indices = [
            indices[i * self.samples_per_gpu:(i + 1) * self.samples_per_gpu]
            for i in np.random.permutation(
                range(len(indices) // self.samples_per_gpu))
        ]
        indices = np.concatenate(indices)
        indices = indices.astype(np.int64).tolist()
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples


@SAMPLERS.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Args:
        dataset: Dataset used for sampling.
        seed (int, Optional): The seed. Default to 0.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 seed=0,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                indice = indice[list(torch.randperm(int(size),
                                                    generator=g))].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


@SAMPLERS.register_module()
class DistributedGivenIterationSampler(Sampler):

    def __init__(self,
                 dataset,
                 total_iter,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 last_iter=-1):
        rank, world_size = get_dist_info()
        assert rank < world_size
        self.dataset = dataset
        self.total_iter = total_iter
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.last_iter = last_iter

        self.total_size = self.total_iter * self.batch_size

        self.indices = self.gen_new_list()

    def __iter__(self):
        return iter(self.indices[(self.last_iter + 1) * self.batch_size:])

    def set_uniform_indices(self, labels, num_classes):
        np.random.seed(0)
        assert (len(labels) == len(self.dataset))
        N = len(labels)
        size_per_label = int(N / num_classes) + 1
        indices = []
        images_lists = [[] for i in range(num_classes)]
        for i, l in enumerate(labels):
            images_lists[l].append(i)
        for i, l in enumerate(images_lists):
            if len(l) == 0:
                continue
            indices.extend(
                np.random.choice(
                    l, size_per_label, replace=(len(l) <= size_per_label)))
        indices = np.array(indices)
        np.random.shuffle(indices)
        indices = indices[:N].astype(np.int)
        # repeat
        all_size = self.total_size * self.world_size
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]
        np.random.shuffle(indices)
        # slice
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]
        assert len(indices) == self.total_size
        # set
        self.indices = indices

    def gen_new_list(self):
        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(0)

        all_size = self.total_size * self.world_size
        indices = np.arange(len(self.dataset))
        indices = indices[:all_size]
        num_repeat = (all_size - 1) // indices.shape[0] + 1
        indices = np.tile(indices, num_repeat)
        indices = indices[:all_size]

        np.random.shuffle(indices)
        beg = self.total_size * self.rank
        indices = indices[beg:beg + self.total_size]

        assert len(indices) == self.total_size

        return indices

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by dataloader
        # return self.total_size - (self.last_iter+1)*self.batch_size
        return self.total_size

    def set_epoch(self, epoch):
        pass


@SAMPLERS.register_module()
class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed,
    with repeated augmentation.
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self,
                 dataset,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 num_repeats: int = 3,
                 **kwargs):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    'Requires distributed package to be available')
            rank = dist.get_rank()
        if num_repeats < 1:
            raise ValueError('num_repeats should be greater than 0')
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.num_samples = int(
            math.ceil(
                len(self.dataset) * self.num_repeats / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # self.num_selected_samples = int(math.ceil(len(self.dataset) / self.num_replicas))
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(start=0, end=len(self.dataset))

        # add extra samples to make it evenly divisible
        indices = torch.repeat_interleave(
            indices, repeats=self.num_repeats, dim=0).tolist()
        padding_size: int = self.total_size - len(indices)
        if padding_size > 0:
            indices += indices[:padding_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
