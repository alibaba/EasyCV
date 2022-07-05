# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import argparse
import os.path as osp
import time
import os

import requests
import torch
try:
    from nni.compression.pytorch import ModelSpeedup
except ImportError:
    raise ImportError(
        'Please read docs and run "pip install https://pai-nni.oss-cn-zhangjiakou.aliyuncs.com/release/2.5/pai_nni-2.5-py3-none-manylinux1_x86_64.whl" '
        'to install pai_nni')

from easycv.models import build_model
from easycv.apis import set_random_seed, train_model, build_optimizer
from easycv.apis.train_misc import build_yolo_optimizer
from easycv.datasets import build_dataloader, build_dataset
from easycv.utils.logger import get_root_logger
from easycv.file import io
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import (CONFIG_TEMPLATE_ZOO,
                                       mmcv_config_fromfile, rebuild_config)
from easycv.utils.dist_utils import get_num_gpu_per_node
from easycv.toolkit.prune.prune_utils import get_prune_layer, load_pruner


def parse_args():
    parser = argparse.ArgumentParser(description='EasyCV prune a model')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save pruned models')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed prune training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help=
        'parameterize param when user specific choose a model config template like YOLOX_EDGE'
    )
    parser.add_argument(
        '--pruning_class',
        type=str,
        default=None,
        help='pruning class for pruning models')
    parser.add_argument(
        '--pruning_algorithm',
        type=str,
        default=None,
        help='pruning algorithm using by pruning class')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--user_config_params',
        nargs=argparse.REMAINDER,
        default=None,
        help='modify config options using the command-line')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.model_type is not None and args.config is None:
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

    cfg = mmcv_config_fromfile(args.config)

    if args.user_config_params is not None:
        assert args.model_type is not None, 'model_type must be setted'
        # rebuild config by user config params
        cfg = rebuild_config(cfg, args.user_config_params)

    # check oss_config and init oss io
    if cfg.get('oss_io_config', None) is not None:
        io.access_oss(**cfg.oss_io_config)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # if `work_dir` is oss path, redirect `work_dir` to local path, add `oss_work_dir` point to oss path,
    # and use osssync hook to upload log and ckpt in work_dir to oss_work_dir
    if cfg.work_dir.startswith('oss://'):
        cfg.oss_work_dir = cfg.work_dir
        cfg.work_dir = osp.join('work_dirs',
                                cfg.work_dir.replace('oss://', ''))
    else:
        cfg.oss_work_dir = None

    # create work_dir
    if not io.exists(cfg.work_dir):
        io.makedirs(cfg.work_dir)

    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    if args.model_type is not None:
        model_name = args.model_type
    else:
        model_name = cfg.model.type

    dummy_input = torch.randn([1, 3, cfg.img_scale[0],
                               cfg.img_scale[1]]).to('cuda')
    prune_layer_names = get_prune_layer(cfg.model.type)

    # set random seeds
    if args.seed is not None:
        logger.info('Set random seed to {}, deterministic: {}'.format(
            args.seed, args.deterministic))
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed

    cfg.gpus = args.gpus

    model = build_model(cfg.model)

    checkpoint = load_checkpoint(model, args.checkpoint)

    if cfg.model.type == 'YOLOX_EDGE' or 'YOLOX':
        prune_optimizer = build_yolo_optimizer(model, cfg.optimizer)
    else:
        prune_optimizer = build_optimizer(model, cfg.optimizer)

    model_layers = []
    for name, module in model.named_modules():
        model_layers.append(name)
    print(model_layers)

    meta = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    distributed = False
    # build dataloader
    shuffle = cfg.data.train.pop('shuffle', True)
    print(f'data shuffle: {shuffle}')

    datasets = [build_dataset(cfg.data.train)]
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=distributed,
            shuffle=shuffle,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=cfg.seed,
            drop_last=getattr(cfg.data, 'drop_last', False),
            reuse_worker_cache=cfg.data.get('reuse_worker_cache', False))
        for ds in datasets
    ]

    iters = datasets[0].num_samples * cfg.total_epochs / \
        (get_num_gpu_per_node() * cfg.data.imgs_per_gpu)
    assert int(
        iters
    ) >= 1200, 'pruner need iters larger than 1200, please increase epoch'

    pruner_config = [{
        'sparsity': 0.5,
        'frequency': 200,
        'start_iter': 0,
        'end_iter': 1200,
        'op_names': prune_layer_names
    }]

    pruner = load_pruner(model, pruner_config, prune_optimizer,
                         args.pruning_class, args.pruning_algorithm)

    train_model(
        model,
        data_loaders,
        cfg,
        distributed=distributed,
        timestamp=timestamp,
        meta=meta)

    # export mask model
    mask_path = osp.join(cfg.work_dir, model_name + '_mask.pth')
    model_path = osp.join(cfg.work_dir, model_name + '_model.pth')
    model.head.decode_in_inference = False
    pruner.export_model(model_path, mask_path)

    model = build_model(cfg.model)
    model.load_state_dict(torch.load(model_path))
    model.head.decode_in_inference = False
    model.cuda()
    model.eval()

    m_speedup = ModelSpeedup(model, dummy_input, mask_path, 'cuda')
    m_speedup.speedup_model()

    # eport prune model
    model.head.decode_in_inference = False
    traced_model_path = osp.join(cfg.work_dir, model_name + '_traced_model.pt')
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, traced_model_path)

    if cfg.oss_work_dir is not None:
        export_oss_path = os.path.join(cfg.oss_work_dir, 'prune_model.pt')
        if not os.path.exists(traced_model_path):
            logger.warning(f'{traced_model_path} does not exists, skip upload')
        else:
            logger.info(f'upload {traced_model_path} to {export_oss_path}')
            io.safe_copy(traced_model_path, export_oss_path)


if __name__ == '__main__':
    main()
