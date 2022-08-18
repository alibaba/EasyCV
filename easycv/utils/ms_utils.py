# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import jsonplus

from easycv.file import io
from easycv.utils.config_tools import Config

MODELSCOPE_PREFIX = 'modelscope'


def to_ms_config(cfg, dump=True, save_path=None):
    """Convert EasyCV config to ModelScope style."""
    # TODO: support multi eval_pipelines

    if isinstance(cfg, str):
        easycv_cfg = Config.fromfile(cfg)
        if dump and save_path is None:
            save_dir = os.path.dirname(cfg)
            save_name = MODELSCOPE_PREFIX + '_' + os.path.splitext(
                os.path.basename(cfg))[0] + '.json'
            save_path = os.path.join(save_dir, save_name)
    else:
        easycv_cfg = cfg
        if dump and save_path is None:
            raise ValueError('Please provide `save_path`!')

    optimizer_options = easycv_cfg.optimizer_config
    optimizer_options.update({'loss_keys': 'total_loss'})

    val_dataset_cfg = easycv_cfg.data.val
    val_imgs_per_gpu = val_dataset_cfg.pop('imgs_per_gpu',
                                           easycv_cfg.data.imgs_per_gpu)
    val_workers_per_gpu = val_dataset_cfg.pop('workers_per_gpu',
                                              easycv_cfg.data.workers_per_gpu)

    log_config = easycv_cfg.log_config

    hooks = [{
        'type': 'CheckpointHook',
        'interval': easycv_cfg.checkpoint_config.interval
    }, {
        'type': 'EvaluationHook',
        'interval': easycv_cfg.eval_config.interval
    }, {
        'type': 'AddLrLogHook'
    }, {
        'type': 'IterTimerHook'
    }]

    custom_hooks = easycv_cfg.get('custom_hooks', [])
    hooks.extend(custom_hooks)

    for log_hook_i in log_config.hooks:
        if log_hook_i['type'] == 'TensorboardLoggerHook':
            # replace with modelscope api
            hooks.append({
                'type': 'TensorboardHook',
                'interval': log_config.interval
            })
        elif log_hook_i['type'] == 'TextLoggerHook':
            # use modelscope api
            hooks.append({
                'type': 'TextLoggerHook',
                'interval': log_config.interval
            })
        else:
            log_hook_i.update({'interval': log_config.interval})
            hooks.append(log_hook_i)

    ms_cfg = Config(
        dict(
            model=easycv_cfg.model,
            dataset=dict(train=easycv_cfg.data.train, val=val_dataset_cfg),
            train=dict(
                work_dir=easycv_cfg.get('work_dir', None),
                max_epochs=easycv_cfg.total_epochs,
                dataloader=dict(
                    batch_size_per_gpu=easycv_cfg.data.imgs_per_gpu,
                    workers_per_gpu=easycv_cfg.data.workers_per_gpu,
                ),
                optimizer=dict(
                    **easycv_cfg.optimizer, options=optimizer_options),
                lr_scheduler=easycv_cfg.lr_config,
                hooks=hooks),
            evaluation=dict(
                dataloader=dict(
                    batch_size_per_gpu=val_imgs_per_gpu,
                    workers_per_gpu=val_workers_per_gpu,
                ),
                metrics={
                    'type': 'EasyCVMetric',
                    'evaluators': easycv_cfg.eval_pipelines[0].evaluators
                })))

    if dump:
        with io.open(save_path, 'w') as f:
            res = jsonplus.dumps(ms_cfg._cfg_dict.to_dict())
            f.write(res)

    return ms_cfg
