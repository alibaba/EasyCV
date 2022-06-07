# Copyright (c) Alibaba, Inc. and its affiliates.
from distutils.version import LooseVersion

import torch

from .best_ckpt_saver_hook import BestCkptSaverHook
from .builder import build_hook
from .byol_hook import BYOLHook
from .dino_hook import DINOHook
from .ema_hook import EMAHook
from .eval_hook import DistEvalHook, EvalHook
from .export_hook import ExportHook
from .extractor import Extractor
from .optimizer_hook import OptimizerHook
from .oss_sync_hook import OSSSyncHook
from .registry import HOOKS
from .show_time_hook import TIMEHook
from .swav_hook import SWAVHook
from .sync_norm_hook import SyncNormHook
from .sync_random_size_hook import SyncRandomSizeHook
from .tensorboard import TensorboardLoggerHookV2
from .wandb import WandbLoggerHookV2
from .yolox_lr_hook import YOLOXLrUpdaterHook
from .yolox_mode_switch_hook import YOLOXModeSwitchHook

__all__ = [
    'BestCkptSaverHook', 'build_hook', 'BYOLHook', 'DINOHook', 'EMAHook',
    'DistEvalHook', 'EvalHook', 'ExportHook', 'Extractor', 'OptimizerHook',
    'OSSSyncHook', 'HOOKS', 'TIMEHook', 'SWAVHook', 'SyncNormHook',
    'SyncRandomSizeHook', 'TensorboardLoggerHookV2', 'WandbLoggerHookV2',
    'YOLOXLrUpdaterHook', 'YOLOXModeSwitchHook'
]

if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
    from .optimizer_hook import AMPFP16OptimizerHook
    __all__.append('AMPFP16OptimizerHook')
