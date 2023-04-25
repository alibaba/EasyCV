# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import jsonplus

from easycv.file import io
from easycv.utils.config_tools import Config

MODELSCOPE_PREFIX = 'modelscope'


class EasyCVMeta:
    ARCH = '__easycv_arch__'

    META = '__easycv_meta__'
    RESERVED_KEYS = 'reserved_keys'


def to_ms_config(cfg,
                 task,
                 ms_model_name,
                 pipeline_name,
                 save_path=None,
                 reserved_keys=[],
                 dump=True):
    """Convert EasyCV config to ModelScope style.

    Args:
        cfg (str | Config): Easycv config file or Config object.
        task (str): Task name in modelscope, refer to: modelscope.utils.constant.Tasks.
        ms_model_name (str): Model name registered in modelscope, model type will be replaced with `ms_model_name`, used in modelscope.
        pipeline_name (str): Predict pipeline name registered in modelscope, refer to: modelscope/pipelines/cv/easycv_pipelines.
        save_path (str): Save path for saving the generated modelscope configuration file. Only valid when dump is True.
        reserved_keys (list of str): Keys conversion may loss some of the original global keys, not all keys will be retained.
            If you need to keep some keys, for example, keep the `CLASSES` key of config for inference, you can specify: reserved_keys=['CLASSES'].
        dump (bool): Whether dump the converted config to `save_path`.
    """
    # TODO: support multi eval_pipelines
    # TODO: support for adding customized required keys to the configuration file

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

    assert save_path.endswith('json'), 'Only support json file!'
    optimizer_options = easycv_cfg.optimizer_config

    val_dataset_cfg = easycv_cfg.data.val
    val_imgs_per_gpu = val_dataset_cfg.pop('imgs_per_gpu',
                                           easycv_cfg.data.imgs_per_gpu)
    val_workers_per_gpu = val_dataset_cfg.pop('workers_per_gpu',
                                              easycv_cfg.data.workers_per_gpu)

    log_config = easycv_cfg.log_config
    predict_config = easycv_cfg.get('predict', None)

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

    ori_model_type = easycv_cfg.model.pop('type')

    ms_cfg = Config(
        dict(
            task=task,
            framework='pytorch',
            plugins=['pai-easycv'],
            preprocessor={},  # adapt to modelscope, do nothing
            model={
                'type': ms_model_name,
                **easycv_cfg.model, EasyCVMeta.ARCH: {
                    'type': ori_model_type
                }
            },
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
                }),
            pipeline=dict(type=pipeline_name, predictor_config=predict_config),
        ))

    for key in reserved_keys:
        ms_cfg.merge_from_dict({key: getattr(easycv_cfg, key)})

    if len(reserved_keys) > 1:
        ms_cfg.merge_from_dict(
            {EasyCVMeta.META: {
                EasyCVMeta.RESERVED_KEYS: reserved_keys
            }})

    if dump:
        with io.open(save_path, 'w') as f:
            res = jsonplus.dumps(
                ms_cfg._cfg_dict.to_dict(), indent=4, sort_keys=False)
            f.write(res)

    return ms_cfg
