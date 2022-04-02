# Copyright (c) Alibaba, Inc. and its affiliates.
import argparse
import json
import os
import os.path as osp
import random
import time

import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from easycv.datasets import build_dataloader, build_dataset
from easycv.file import io
from easycv.models import build_model
from easycv.utils import (dist_forward_collect, get_root_logger,
                          nondist_forward_collect)
from easycv.utils.config_tools import mmcv_config_fromfile


def set_random_seed(seed, deterministic=True):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExtractProcess(object):

    def __init__(self, extract_list=['backbone']):
        self.extract_list = extract_list
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def _forward_func(self, model, **kwargs):
        if hasattr(model.module, 'update_extract_list'):
            for k in self.extract_list:
                model.module.update_extract_list(k)

        feats = model(mode='extract', **kwargs)

        for k in self.extract_list:
            if type(feats[k]) is torch.Tensor:
                feats[k] = [feats[k]]

        flat_feats = []
        for feat in feats['backbone']:
            if len(feat.shape) > 2:
                feat = self.pool(feat)
            flat_feats.append(feat.view(feat.size(0), -1))

        feat_dict = {
            'feat{}'.format(i + 1): feat.cpu()
            for i, feat in enumerate(flat_feats)
        }

        if 'label' in kwargs.keys():
            feat_dict['label'] = kwargs['label']
        if 'gt_label' in kwargs.keys():
            feat_dict['label'] = kwargs['gt_label']
        return feat_dict

    def extract(self, model, data_loader, distributed=False):
        model.eval()
        func = lambda **x: self._forward_func(model, **x)

        if hasattr(data_loader, 'dataset'):
            length = len(data_loader.dataset)
        else:
            length = data_loader.data_length

        if distributed:
            rank, world_size = get_dist_info()
            results = dist_forward_collect(func, data_loader, rank, length)
        else:
            results = nondist_forward_collect(func, data_loader, length)
        return results


def parse_args():
    parser = argparse.ArgumentParser(
        description='EVTORCH batch（use dataloader） extract features of a model'
    )
    parser.add_argument(
        'config', help='config file path', type=str, default=None)
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--pretrained',
        default='random',
        help='pretrained model file, exclusive to --checkpoint')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save logs and models')
    parser.add_argument(
        '--extract_list',
        type=list,
        default=['backbone'],
        help='the dir to save logs and models')
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
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    # set cudnn_benchmark
    cfg = mmcv_config_fromfile(args.config)

    if cfg.get('oss_io_config', None):
        io.access_oss(**cfg.oss_io_config)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # update configs according to CLI args
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    # checkpoint and pretrained are exclusive
    assert args.pretrained == 'random' or args.checkpoint is None, \
        'Checkpoint and pretrained are exclusive.'

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        if args.launcher == 'slurm':
            cfg.dist_params['port'] = args.port
        init_dist(args.launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, 'extract_{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    datasets = [build_dataset(cfg.data.extract)]
    seed = 0
    set_random_seed(seed)
    data_loader = [
        build_dataloader(
            ds,
            cfg.data.imgs_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus,
            dist=distributed,
            shuffle=False,
            replace=getattr(cfg.data, 'sampling_replace', False),
            seed=seed,
            drop_last=getattr(cfg.data, 'drop_last', False)) for ds in datasets
    ]

    # specify pretrained model
    if args.pretrained != 'random':
        assert isinstance(args.pretrained, str)
        cfg.model.pretrained = args.pretrained

    assert os.path.exists(args.checkpoint), \
        'checkpoint must be set when use extractor!'
    ckpt_meta = torch.load(args.checkpoint).get('meta', None)
    cfg_model = cfg.get('model', None)

    if cfg_model is not None:
        logger.info('load model scripts from cfg config')
        model = build_model(cfg_model)
    else:
        assert ckpt_meta is not None, 'extract need either cfg model or ckpt with meta!'
        logger.info('load model scripts from ckpt meta')
        ckpt_cfg = json.loads(ckpt_meta['config'])
        if 'model' not in ckpt_cfg:
            raise ValueError(
                'build model from %s, must use model after export' %
                (args.checkpoint))
        model = build_model(ckpt_cfg['model'])

    # build the model and load checkpoint

    if args.checkpoint is not None:
        logger.info('Use checkpoint: {} to extract features'.format(
            args.checkpoint))
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    elif args.pretrained != 'random':
        logger.info('Use pretrained model: {} to extract features'.format(
            args.pretrained))
    else:
        logger.info('No checkpoint or pretrained is give, use random init.')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    # build extraction processor
    extractor = ExtractProcess(extract_list=args.extract_list)

    # run
    outputs = extractor.extract(model, data_loader[0], distributed=distributed)

    rank, _ = get_dist_info()
    mmcv.mkdir_or_exist(args.work_dir)

    if rank == 0:
        for key, val in outputs.items():
            split_num = len(cfg.split_name)
            split_at = cfg.split_at
            print(split_num, split_at)
            for ss in range(split_num):
                output_file = '{}/{}_{}.npy'.format(args.work_dir,
                                                    cfg.split_name[ss], key)
                if ss == 0:
                    np.save(output_file, val[:split_at[0]])
                elif ss == split_num - 1:
                    np.save(output_file, val[split_at[-1]:])
                else:
                    np.save(output_file, val[split_at[ss - 1]:split_at[ss]])


if __name__ == '__main__':
    main()
