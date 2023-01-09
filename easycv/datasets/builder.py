# Copyright (c) Alibaba, Inc. and its affiliates.
import copy

from easycv.datasets.shared.dataset_wrappers import (ConcatDataset,
                                                     RepeatDataset)
from easycv.utils.registry import build_from_cfg
from .registry import DALIDATASETS, DATASETS, DATASOURCES, SAMPLERS


def _concat_dataset(cfg, default_args=None):
    ann_files = cfg['data_source']['ann_file']
    img_prefixes = cfg['data_source'].get('img_prefix', None)
    seg_prefixes = cfg['data_source'].get('seg_prefix', None)
    proposal_files = cfg['data_source'].get('proposal_file', None)

    datasets = []
    num_dset = len(ann_files)
    for i in range(num_dset):
        data_cfg = copy.deepcopy(cfg)
        data_cfg['data_source']['ann_file'] = ann_files[i]
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg['data_source']['img_prefix'] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg['data_source']['seg_prefix'] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg['data_source']['proposal_file'] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif isinstance(cfg['data_source'].get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset


def build_dali_dataset(cfg, default_args=None):
    return build_from_cfg(cfg, DALIDATASETS, default_args)


def build_datasource(cfg):
    return build_from_cfg(cfg, DATASOURCES)


def build_sampler(cfg, default_args=None):
    return build_from_cfg(cfg, SAMPLERS, default_args)
