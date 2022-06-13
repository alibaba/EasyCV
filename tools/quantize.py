# Copyright (c) Alibaba, Inc. and its affiliates.
"""
isort:skip_file
"""
import argparse
import os
import os.path as osp
import sys
import time

import requests
import torch
try:
    from blade_compression.fx_quantization.prepare import (convert,
                                                           enable_calibration,
                                                           prepare_fx)
except ImportError:
    raise ImportError(
        'Please read docs and run "pip install http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/third_party/blade_compression-0.0.1-py3-none-any.whl" '
        'to install blade_compression')

from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info

from easycv.models import build_model
from easycv.apis import single_cpu_test, single_gpu_test
from easycv.core.evaluation.builder import build_evaluator
from easycv.datasets import build_dataloader, build_dataset
from easycv.utils.logger import get_root_logger
from easycv.utils.flops_counter import get_model_info
from easycv.file import io
from easycv.toolkit.quantize.quantize_utils import calib, quantize_config_check
from easycv.utils.checkpoint import load_checkpoint
from easycv.utils.config_tools import (CONFIG_TEMPLATE_ZOO,
                                       mmcv_config_fromfile, rebuild_config)

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(
    os.path.abspath(
        osp.join(os.path.dirname(os.path.dirname(__file__)), '../')))


def parse_args():
    parser = argparse.ArgumentParser(description='EasyCV quantize a model')
    parser.add_argument('config', help='model config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work_dir',
        type=str,
        default=None,
        help='the dir to save quantized models')
    parser.add_argument(
        '--model_type',
        type=str,
        default=None,
        help=
        'parameterize param when user specific choose a model config template like CLASSIFICATION: classification.py'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='the device quantized models use')
    parser.add_argument(
        '--backend',
        type=str,
        default='PyTorch',
        help="the quantized models's framework")
    parser.add_argument(
        '--user_config_params',
        nargs=argparse.REMAINDER,
        default=None,
        help='modify config options using the command-line')
    args = parser.parse_args()
    return args


def quantize_eval(cfg, model, eval_mode):
    for eval_pipe in cfg.eval_pipelines:
        eval_data = eval_pipe.data
        # build the dataloader
        imgs_per_gpu = eval_data.pop('imgs_per_gpu', cfg.data.imgs_per_gpu)

        dataset = build_dataset(eval_data)
        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=imgs_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=False,
            shuffle=False)

        if eval_mode == 'cuda':
            outputs = single_gpu_test(model, data_loader, mode=eval_pipe.mode)
        elif eval_mode == 'cpu':
            outputs = single_cpu_test(model, data_loader, mode=eval_pipe.mode)

        rank, _ = get_dist_info()
        if rank == 0:
            for t in eval_pipe.evaluators:
                if 'metric_type' in t:
                    t.pop('metric_type')
            evaluators = build_evaluator(eval_pipe.evaluators)
            eval_result = dataset.evaluate(outputs, evaluators=evaluators)
            print(f'\n eval_result {eval_result}')


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

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, '{}.log'.format(timestamp))
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    cfg.model.pretrained = None
    if cfg.model.get('neck'):
        if type(cfg.model.neck) is list:
            pass
        else:
            if cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

    # build the model and load checkpoint
    model = build_model(cfg.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    logger.info(f'use device {device}')
    checkpoint = load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()
    model.to(device)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'meta' in checkpoint and 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    elif hasattr(cfg, 'CLASSES'):
        model.CLASSES = cfg.CLASSES

    # MMDataParallel for gpu
    if device == 'cuda':
        base_model = MMDataParallel(model, device_ids=[0])
    else:
        base_model = model

    # eval base model before quantizing
    get_model_info(model, cfg.img_scale, cfg.model, logger)
    quantize_eval(cfg, base_model, device)

    # setting quantize config
    quantize_config = quantize_config_check(args.device, args.backend,
                                            args.model_type)

    model.to('cuda')
    prepared_backbone = prepare_fx(model.backbone.eval(), quantize_config)
    enable_calibration(prepared_backbone)
    # build calib dataloader, only need 50 samples
    logger.info('build calib dataloader')
    eval_data = cfg.eval_pipelines[0].data
    imgs_per_gpu = eval_data.pop('imgs_per_gpu', cfg.data.imgs_per_gpu)

    dataset = build_dataset(eval_data)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=imgs_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # guarantee accuracy
    logger.info('guarantee calib')
    calib(prepared_backbone, data_loader)

    # quantized model on cpu
    model.to('cpu')

    # quantizing model
    logger.info('convert model')
    quantized_backbone, _ = convert(prepared_backbone, quantize_config)
    model.backbone = quantized_backbone
    model.eval()

    # cpu eval
    logger.info('quantized model eval')
    get_model_info(model, cfg.img_scale, cfg.model, logger)
    quantize_eval(cfg, model, 'cpu')

    input_shape = (1, 3, cfg.img_scale[0], cfg.img_scale[1])
    model.head.decode_in_inference = False
    dummy = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, dummy)
    model_path = osp.join(cfg.work_dir, 'quantize_model.pt')
    torch.jit.save(traced_model, model_path)

    if cfg.oss_work_dir is not None:
        export_oss_path = os.path.join(cfg.oss_work_dir, 'quantize_model.pt')
        if not os.path.exists(model_path):
            logger.warning(f'{model_path} does not exists, skip upload')
        else:
            logger.info(f'upload {model_path} to {export_oss_path}')
            io.safe_copy(model_path, export_oss_path)


if __name__ == '__main__':
    main()
