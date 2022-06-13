# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import time
import json
import argparse
import os
import os.path as osp
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(
    os.path.abspath(
        osp.join(os.path.dirname(os.path.dirname(__file__)), '../')))

import mmcv
import requests
import torch
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist

from easycv import datasets
from easycv.apis import multi_gpu_test, single_gpu_test
from easycv.core.evaluation.builder import build_evaluator
from easycv.datasets import build_dataloader, build_dataset
from easycv.file import io
from easycv.models import build_model
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import (CONFIG_TEMPLATE_ZOO,
                                       mmcv_config_fromfile, rebuild_config)
from easycv.utils.mmlab_utils import dynamic_adapt_for_mmlab
from easycv.utils.setup_env import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(
        description='EasyCV test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir', help='the directory to save evaluation logs')
    parser.add_argument('--out', help='output result file in pickle format')
    # parser.add_argument(
    #     '--fuse-conv-bn',
    #     action='store_true',
    #     help='Whether to fuse conv and bn, this will slightly increase'
    #     'the inference speed')
    parser.add_argument(
        '--inference-only',
        action='store_true',
        help='save the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument('--eval', action='store_true', help='evaluate result')
    parser.add_argument('--fp16', action='store_true', help='use fp16')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    # parser.add_argument(
    #     '--show-score-thr',
    #     type=float,
    #     default=0.3,
    #     help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help=
        'parameterize param when user specific choose a model config template like CLASSIFICATION: classification.py'
    )
    parser.add_argument(
        '--user_config_params',
        nargs=argparse.REMAINDER,
        default=None,
        help='modify config options using the command-line')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.eval or args.inference_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--inference-only", "--show" or "--show-dir"')

    if args.eval and args.inference_only:
        raise ValueError(
            '--eval and --inference_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

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

    cfg = mmcv_config_fromfile(args.config)

    if args.user_config_params is not None:
        assert args.model_type is not None, 'model_type must be setted'
        # rebuild config by user config params
        cfg = rebuild_config(cfg, args.user_config_params)

    # check oss_config and init oss io
    if cfg.get('oss_io_config', None) is not None:
        io.access_oss(**cfg.oss_io_config)

    # set multi-process settings
    setup_multi_processes(cfg)

    # dynamic adapt mmdet models
    dynamic_adapt_for_mmlab(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if type(cfg.model.neck) is list:
            pass
        else:
            if cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None
    # cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()

    if args.work_dir is not None and rank == 0:
        if not io.exists(args.work_dir):
            io.makedirs(args.work_dir)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join(args.work_dir, 'eval_{}.json'.format(timestamp))

    # build the model and load checkpoint
    model = build_model(cfg.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'use device {device}')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)
    model.to(device)
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(cfg, 'CLASSES'):
        model.CLASSES = cfg.CLASSES

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)

    assert 'eval_pipelines' in cfg, 'eval_pipelines is needed for testting'
    for eval_pipe in cfg.eval_pipelines:
        eval_data = eval_pipe.get('data', None) or cfg.data.val
        # build the dataloader
        if eval_data.get('dali', False):
            data_loader = datasets.build_dali_dataset(
                eval_data).get_dataloader()
            # dali dataloader implements `evaluate` func, so use it as dummy dataset
            dataset = data_loader
        else:
            # dataset does not need imgs_per_gpu, except dali dataset
            imgs_per_gpu = eval_data.pop('imgs_per_gpu', cfg.data.imgs_per_gpu)

            dataset = build_dataset(eval_data)
            data_loader = build_dataloader(
                dataset,
                imgs_per_gpu=imgs_per_gpu,
                workers_per_gpu=cfg.data.workers_per_gpu,
                dist=distributed,
                shuffle=False)

        if not distributed:
            outputs = single_gpu_test(
                model, data_loader, mode=eval_pipe.mode, use_fp16=args.fp16)
        else:
            outputs = multi_gpu_test(
                model,
                data_loader,
                mode=eval_pipe.mode,
                tmp_dir=args.tmpdir,
                gpu_collect=args.gpu_collect,
                use_fp16=args.fp16)

        if rank == 0:
            if args.out:
                print(f'\nwriting results to {args.out}')
                mmcv.dump(outputs, args.out)
            eval_kwargs = {}
            if args.options is not None:
                eval_kwargs.update(args.options)

            if args.inference_only:
                raise RuntimeError('not implemented')
                dataset.format_results(outputs, **eval_kwargs)
            if args.eval:
                for t in eval_pipe.evaluators:
                    if 'metric_type' in t:
                        t.pop('metric_type')
                evaluators = build_evaluator(eval_pipe.evaluators)
                eval_result = dataset.evaluate(outputs, evaluators=evaluators)
                print(f'\n eval_result {eval_result}')
                if args.work_dir is not None:
                    with io.open(log_file, 'w') as f:
                        json.dump(eval_result, f)


if __name__ == '__main__':
    main()
