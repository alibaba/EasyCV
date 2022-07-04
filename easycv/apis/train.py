# Copyright (c) Alibaba, Inc. and its affiliates.
import random
import re
from distutils.version import LooseVersion

import numpy as np
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import DistSamplerSeedHook, obj_from_dict
from torch import optim

from easycv.apis.train_misc import build_yolo_optimizer
from easycv.core import optimizer
from easycv.core.evaluation.builder import build_evaluator
from easycv.core.evaluation.metric_registry import METRICS
from easycv.datasets import build_dataloader, build_dataset
from easycv.datasets.utils import is_dali_dataset_type
from easycv.hooks import (BestCkptSaverHook, DistEvalHook, EMAHook, EvalHook,
                          ExportHook, OptimizerHook, OSSSyncHook, build_hook)
from easycv.hooks.optimizer_hook import AMPFP16OptimizerHook
from easycv.runner import EVRunner
from easycv.utils.eval_utils import generate_best_metric_name
from easycv.utils.logger import get_root_logger, print_log


def set_random_seed(seed, deterministic=False):
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
        if hasattr(torch, '_set_deterministic'):
            torch._set_deterministic(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(model,
                data_loaders,
                cfg,
                distributed=False,
                timestamp=None,
                meta=None,
                use_fp16=False,
                validate=True,
                gpu_collect=True):
    """ Training API.

    Args:
        model (:obj:`nn.Module`): user defined model
        data_loaders: a list of dataloader for training data
        cfg: config object
        distributed: distributed training or not
        timestamp: time str formated as '%Y%m%d_%H%M%S'
        meta: a dict containing meta data info, such as env_info, seed, iter, epoch
        use_fp16: use fp16 training or not
        validate: do evaluation while training
        gpu_collect: use gpu collect or cpu collect for tensor gathering

    """
    logger = get_root_logger(cfg.log_level)
    print('GPU INFO : ', torch.cuda.get_device_name(0))

    if cfg.model.type == 'YOLOX':
        optimizer = build_yolo_optimizer(model, cfg.optimizer)
    else:
        optimizer = build_optimizer(model, cfg.optimizer)

    # when use amp from apex, we should initialze amp with model not wrapper by DDP or DP,
    # so  we need to inialize amp here. In torch 1.6 or later, we do not need this
    if use_fp16 and LooseVersion(torch.__version__) < LooseVersion('1.6.0'):
        from apex import amp
        model, optimizer = amp.initialize(
            model.to('cuda'), optimizer, opt_level='O1')

    # SyncBatchNorm
    open_sync_bn = cfg.get('sync_bn', False)
    if open_sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to('cuda')
        logger.info('Using SyncBatchNorm()')

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            find_unused_parameters=find_unused_parameters,
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    else:
        model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    runner = EVRunner(
        model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta,
        fp16_enable=use_fp16)
    runner.data_loader = data_loaders

    # an ugly walkaround to make the .log and .log.json filenames the same
    runner.timestamp = timestamp
    optimizer_config = cfg.optimizer_config

    if use_fp16:
        assert torch.cuda.is_available(), 'cuda is needed for fp16'
        optimizer_config = AMPFP16OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)

    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    if distributed:
        logger.info('register DistSamplerSeedHook')
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    validate = False
    if 'eval_pipelines' in cfg:
        if isinstance(cfg.eval_pipelines, dict):
            cfg.eval_pipelines = [cfg.eval_pipelines]
        if len(cfg.eval_pipelines) > 0:
            validate = True
            runner.logger.info('open validate hook')

    best_metric_name = [
    ]  # default is eval_pipe.evaluators[0]['type'] + eval_dataset_name + [metric_names]
    best_metric_type = []
    if validate:
        interval = cfg.eval_config.pop('interval', 1)
        for idx, eval_pipe in enumerate(cfg.eval_pipelines):
            data = eval_pipe.get('data', None) or cfg.data.val
            dist_eval = eval_pipe.get('dist_eval', False)

            evaluator_cfg = eval_pipe.evaluators[0]
            # get the metric_name
            eval_dataset_name = evaluator_cfg.get('dataset_name', None)
            default_metrics = METRICS.get(evaluator_cfg['type'])['metric_name']
            default_metric_type = METRICS.get(
                evaluator_cfg['type'])['metric_cmp_op']
            if 'metric_names' not in evaluator_cfg:
                evaluator_cfg['metric_names'] = default_metrics
            eval_metric_names = evaluator_cfg['metric_names']

            # get the metric_name
            this_metric_names = generate_best_metric_name(
                evaluator_cfg['type'], eval_dataset_name, eval_metric_names)
            best_metric_name = best_metric_name + this_metric_names

            # get the metric_type
            this_metric_type = evaluator_cfg.pop('metric_type',
                                                 default_metric_type)
            this_metric_type = this_metric_type + ['max'] * (
                len(this_metric_names) - len(this_metric_type))
            best_metric_type = best_metric_type + this_metric_type

            imgs_per_gpu = data.pop('imgs_per_gpu', cfg.data.imgs_per_gpu)
            workers_per_gpu = data.pop('workers_per_gpu',
                                       cfg.data.workers_per_gpu)
            if not is_dali_dataset_type(data['type']):
                val_dataset = build_dataset(data)
                val_dataloader = build_dataloader(
                    val_dataset,
                    imgs_per_gpu=imgs_per_gpu,
                    workers_per_gpu=workers_per_gpu,
                    dist=(distributed and dist_eval),
                    shuffle=False,
                    seed=cfg.seed)
            else:
                default_args = dict(
                    batch_size=imgs_per_gpu,
                    workers_per_gpu=workers_per_gpu,
                    distributed=distributed)
                val_dataset = build_dataset(data, default_args)
                val_dataloader = val_dataset.get_dataloader()

            evaluators = build_evaluator(eval_pipe.evaluators)
            eval_cfg = cfg.eval_config
            eval_cfg['evaluators'] = evaluators
            eval_hook = DistEvalHook if (distributed
                                         and dist_eval) else EvalHook
            if eval_hook == EvalHook:
                eval_cfg.pop('gpu_collect', None)  # only use in DistEvalHook
            logger.info(f'register EvaluationHook {eval_cfg}')
            # only flush log buffer at the last eval hook
            flush_buffer = (idx == len(cfg.eval_pipelines) - 1)
            runner.register_hook(
                eval_hook(
                    val_dataloader,
                    interval=interval,
                    mode=eval_pipe.mode,
                    flush_buffer=flush_buffer,
                    **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')

            common_params = {}
            if hook_cfg.type == 'DeepClusterHook':
                common_params = dict(
                    dist_mode=distributed, data_loaders=data_loaders)
            else:
                common_params = dict(dist_mode=distributed)

            hook = build_hook(hook_cfg, default_args=common_params)
            runner.register_hook(hook, priority=priority)

    if cfg.get('ema', None):
        runner.logger.info('register ema hook')
        runner.register_hook(EMAHook(decay=cfg.ema.decay))

    if len(best_metric_name) > 0:
        runner.register_hook(
            BestCkptSaverHook(
                by_epoch=True,
                save_optimizer=True,
                best_metric_name=best_metric_name,
                best_metric_type=best_metric_type))

    # export hook
    if getattr(cfg, 'checkpoint_sync_export', False):
        runner.register_hook(ExportHook(cfg))

    # oss sync hook
    if cfg.oss_work_dir is not None:
        if cfg.checkpoint_config.get('by_epoch', True):
            runner.register_hook(
                OSSSyncHook(
                    cfg.work_dir,
                    cfg.oss_work_dir,
                    interval=cfg.checkpoint_config.interval,
                    **cfg.get('oss_sync_config', {})))
        else:
            runner.register_hook(
                OSSSyncHook(
                    cfg.work_dir,
                    cfg.oss_work_dir,
                    interval=1,
                    iter_interval=cfg.checkpoint_config.interval),
                **cfg.get('oss_sync_config', {}))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.logger.info(f'load checkpoint from {cfg.load_from}')
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def get_skip_list_keywords(model):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    return skip, skip_keywords


def _set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        print(name)
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or (name in skip_list) or \
                _check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay}, {'params': no_decay, 'weight_decay': 0.}]


def _check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def build_optimizer(model, optimizer_cfg):
    """Build optimizer from configs.

    Args:
        model (:obj:`nn.Module`): The model with parameters to be optimized.
        optimizer_cfg (dict): The config dict of the optimizer.

            Positional fields are:
                - type: class name of the optimizer.
                - lr: base learning rate.

            Optional fields are:
                - any arguments of the corresponding optimizer type, e.g.,
                  weight_decay, momentum, etc.
                - paramwise_options: a dict with regular expression as keys
                  to match parameter names and a dict containing options as
                  values. Options include 6 fields: lr, lr_mult, momentum,
                  momentum_mult, weight_decay, weight_decay_mult.

    Returns:
        torch.optim.Optimizer: The initialized optimizer.

    Example:
        >>> model = torch.nn.modules.Conv1d(1, 1, 1)
        >>> paramwise_options = {
        >>>     '(bn|gn)(\d+)?.(weight|bias)': dict(weight_decay_mult=0.1),
        >>>     '\Ahead.': dict(lr_mult=10, momentum=0)}
        >>> optimizer_cfg = dict(type='SGD', lr=0.01, momentum=0.9,
        >>>                      weight_decay=0.0001,
        >>>                      paramwise_options=paramwise_options)
        >>> optimizer = build_optimizer(model, optimizer_cfg)
    """

    if hasattr(model, 'module'):
        model = model.module

    # some special model (DINO) only need to optimize parts of parameter,  this kind of model will
    # provide attribute get_params_groups to initial optimizer, as we catch this attribute, we do this if
    if hasattr(model, 'get_params_groups'):
        print('type : ', type(model),
              'trigger opimizer model param_groups set for DINO')
        parameters = model.get_params_groups()
        optimizer_cfg = optimizer_cfg.copy()
        optimizer_cls = getattr(optimizer, optimizer_cfg.pop('type'))
        return optimizer_cls(parameters, **optimizer_cfg)

    # for some model which use transformer(swin/shuffle/cswin), we should set it bias with no weight decay
    set_var_bias_nowd = optimizer_cfg.pop('set_var_bias_nowd', None)
    if set_var_bias_nowd is None:
        set_var_bias_nowd = optimizer_cfg.pop(
            'trans_weight_decay_set', None
        )  # this is failback when we switch version, set_var_bias_nowd used called trans_weight_decay_set
    if set_var_bias_nowd is not None:
        print('type : ', type(model), 'trigger transformer set_var_bias_nowd')
        skip = []
        skip_keywords = []
        assert (type(set_var_bias_nowd) is list)
        for model_part in set_var_bias_nowd:
            mpart = getattr(model, model_part, None)
            if mpart is not None:
                tskip, tskip_keywords = get_skip_list_keywords(mpart)
                skip += tskip
                skip_keywords += tskip_keywords
        parameters = _set_weight_decay(model, skip, skip_keywords)
        optimizer_cfg = optimizer_cfg.copy()
        optimizer_cls = getattr(optimizer, optimizer_cfg.pop('type'))
        return optimizer_cls(parameters, **optimizer_cfg)

    optimizer_cfg = optimizer_cfg.copy()
    paramwise_options = optimizer_cfg.pop('paramwise_options', None)
    # if no paramwise option is specified, just use the global setting

    if paramwise_options is None:
        return obj_from_dict(optimizer_cfg, optimizer,
                             dict(params=model.parameters()))
    else:
        assert isinstance(paramwise_options, dict)
        params = []
        for name, param in model.named_parameters():
            param_group = {'params': [param]}
            if not param.requires_grad:
                params.append(param_group)
                continue

            for regexp, options in paramwise_options.items():
                if re.search(regexp, name):
                    for key, value in options.items():
                        if key.endswith('_mult'):  # is a multiplier
                            key = key[:-5]
                            assert key in optimizer_cfg, \
                                '{} not in optimizer_cfg'.format(key)
                            value = optimizer_cfg[key] * value
                        param_group[key] = value
                        if not dist.is_initialized() or dist.get_rank() == 0:
                            print_log('paramwise_options -- {}: {}={}'.format(
                                name, key, value))

            # otherwise use the global settings
            params.append(param_group)

        optimizer_cls = getattr(optimizer, optimizer_cfg.pop('type'))
        return optimizer_cls(params, **optimizer_cfg)
