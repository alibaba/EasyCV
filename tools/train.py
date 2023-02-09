# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
from __future__ import division
import argparse
import importlib
import json
import os
import os.path as osp
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(
    os.path.abspath(
        osp.join(os.path.dirname(os.path.dirname(__file__)), '../')))

# adapt to torchacc, init before some torch imports
from easycv.utils.torchacc_util import is_torchacc_enabled
if is_torchacc_enabled():
    from easycv.toolkit.torchacc import torchacc_init
    torchacc_init()

import time
import requests
import torch
import torch.distributed as dist
from mmcv.runner import init_dist
from mmcv import DictAction

from easycv import __version__
from easycv.apis import init_random_seed, set_random_seed, train_model
from easycv.datasets import build_dataloader, build_dataset
from easycv.datasets.utils import is_dali_dataset_type
from easycv.file import io
from easycv.models import build_model
from easycv.utils.collect_env import collect_env
from easycv.utils.logger import get_root_logger
from easycv.utils import mmlab_utils
from easycv.utils.config_tools import (traverse_replace, CONFIG_TEMPLATE_ZOO,
                                       mmcv_config_fromfile,
                                       pai_config_fromfile)
from easycv.utils.dist_utils import get_device, is_master
from easycv.utils.setup_env import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument(
        'config', help='train config file path', type=str, default=None)
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--resume_from', help='the checkpoint file to resume from')
    parser.add_argument('--load_from', help='the checkpoint file to load from')
    parser.add_argument(
        '--pretrained', default=None, help='pretrained model file')
    parser.add_argument(
        '--no_validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument('--fp16', action='store_true', help='use fp16')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--port',
        type=int,
        default=29500,
        help='port only works when launcher=="slurm"')
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help=
        'parameterize param when user specific choose a model config template like CLASSIFICATION: classification.py'
    )
    parser.add_argument(
        '--user_config_params',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed. Single quote double quote equivalent.')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    if args.model_type is not None:
        assert args.model_type in CONFIG_TEMPLATE_ZOO, 'model_type must be in [%s]' % (
            ', '.join(CONFIG_TEMPLATE_ZOO.keys()))
        print('model_type=%s, config file will be replaced by %s' %
              (args.model_type, CONFIG_TEMPLATE_ZOO[args.model_type]))
        args.config = CONFIG_TEMPLATE_ZOO[args.model_type]

    if args.config.startswith('http'):

        r = requests.get(args.config)
        # download config in current dir
        tpath = args.config.split('/')[-1]
        while not osp.exists(tpath):
            try:
                with open(tpath, 'wb') as code:
                    code.write(r.content)
            except:
                pass

        args.config = tpath

    # build cfg
    if args.user_config_params is None:
        cfg = mmcv_config_fromfile(args.config)
    else:
        cfg = pai_config_fromfile(args.config, args.user_config_params,
                                  args.model_type)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    if cfg.get('work_dir', None) is None:
        cfg.work_dir = './work_dir'

    # if `work_dir` is oss path, redirect `work_dir` to local path, add `oss_work_dir` point to oss path,
    # and use osssync hook to upload log and ckpt in work_dir to oss_work_dir
    if cfg.work_dir.startswith('oss://'):
        cfg.oss_work_dir = cfg.work_dir
        cfg.work_dir = osp.join('work_dirs',
                                cfg.work_dir.replace('oss://', ''))
    else:
        cfg.oss_work_dir = None

    if args.resume_from is not None and len(args.resume_from) > 0:
        cfg.resume_from = args.resume_from
    if args.load_from is not None and len(args.load_from) > 0:
        cfg.load_from = args.load_from

    # dynamic adapt mmdet models
    mmlab_utils.dynamic_adapt_for_mmlab(cfg)

    cfg.gpus = args.gpus

    # check memcached package exists
    if importlib.util.find_spec('mc') is None:
        traverse_replace(cfg, 'memcached', False)

    # check oss_config and init oss io
    if cfg.get('oss_io_config', None) is not None:
        io.access_oss(**cfg.oss_io_config)
    # init distributed env first, since logger depends on the dist info.
    if not is_torchacc_enabled():
        if args.launcher == 'none':
            assert cfg.model.type not in \
                ['DeepCluster', 'MOCO', 'SimCLR', 'ODC', 'NPID'], \
                '{} does not support non-dist training.'.format(cfg.model.type)
        else:
            if args.launcher == 'slurm':
                cfg.dist_params['port'] = args.port
            init_dist(args.launcher, **cfg.dist_params)

    distributed = torch.cuda.is_available(
    ) and torch.distributed.is_initialized()

    # create work_dir
    if not io.exists(cfg.work_dir):
        io.makedirs(cfg.work_dir)
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([('{}: {}'.format(k, v))
                          for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info('Distributed training: {}'.format(distributed))
    logger.info('Config:\n{}'.format(cfg.text))
    logger.info('Config Dict:\n{}'.format(json.dumps(cfg._cfg_dict)))
    logger.info('GPU INFO : {}'.format(torch.cuda.get_device_name(0)))

    # set random seeds
    # Using different seeds for different ranks may reduce accuracy
    seed = init_random_seed(args.seed, device=get_device())
    seed = seed + dist.get_rank() if args.diff_seed else seed
    if is_torchacc_enabled():
        assert seed is not None, 'Must provide `seed` to sync model initializer if use torchacc!'

    if seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            seed, args.deterministic))
        set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed

    if args.pretrained is not None:
        assert isinstance(args.pretrained, str)
        cfg.model.pretrained = args.pretrained
    model = build_model(cfg.model)

    if is_master():
        print(model)

    if 'stage' in cfg.model and cfg.model['stage'] == 'EDGE':
        from easycv.utils.flops_counter import get_model_info
        get_model_info(model, cfg.img_scale, cfg.model, logger)

    assert len(cfg.workflow) == 1, 'Validation is called by hook.'
    if cfg.checkpoint_config is not None:
        # save easycv version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            easycv_version=__version__, config=cfg.text)

    # build dataloader
    if not is_dali_dataset_type(cfg.data.train['type']):
        shuffle = cfg.data.train.pop('shuffle', True)
        print(f'data shuffle: {shuffle}')

        # for odps data_source
        if cfg.data.train.data_source.type == 'OdpsReader' and cfg.data.train.data_source.get(
                'odps_io_config', None) is None:
            cfg.data.train.data_source['odps_io_config'] = cfg.get(
                'odps_io_config', None)
            assert (
                cfg.data.train.data_source.get('odps_io_config',
                                               None) is not None
            ), 'odps config must be set in cfg file / cfg.data.train.data_source !!'
            shuffle = False

        if getattr(cfg.data, 'pin_memory', False):
            mmlab_utils.fix_dc_pin_memory()
        datasets = [build_dataset(cfg.data.train)]
        data_loaders = [
            build_dataloader(
                ds,
                cfg.data.imgs_per_gpu,
                cfg.data.workers_per_gpu,
                cfg.gpus,
                dist=distributed,
                shuffle=shuffle,
                pin_memory=getattr(cfg.data, 'pin_memory', False),
                replace=getattr(cfg.data, 'sampling_replace', False),
                seed=cfg.seed,
                # The default should be set to True, because sometimes the last batch is not sampled enough, causing an error in batchnorm
                drop_last=getattr(cfg.data, 'drop_last', True),
                reuse_worker_cache=cfg.data.get('reuse_worker_cache', False),
                persistent_workers=cfg.data.get('persistent_workers', False),
                collate_hooks=cfg.data.get('train_collate_hooks', []),
                use_repeated_augment_sampler=cfg.data.get(
                    'use_repeated_augment_sampler', False)) for ds in datasets
        ]
    else:
        default_args = dict(
            batch_size=cfg.data.imgs_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            distributed=distributed)
        dataset = build_dataset(cfg.data.train, default_args)
        data_loaders = [dataset.get_dataloader()]

    # # add an attribute for visualization convenience
    train_model(
        model,
        data_loaders,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta,
        validate=(not args.no_validate),
        use_fp16=args.fp16)


if __name__ == '__main__':
    main()
