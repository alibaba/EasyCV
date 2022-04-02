# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

from mmcv.runner import get_dist_info
from mmcv.runner.hooks import Hook
from torch import nn

from ..utils.dist_utils import all_reduce_dict
from .registry import HOOKS


def get_norm_states(module):
    async_norm_states = OrderedDict()
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._NormBase):
            for k, v in child.state_dict().items():
                async_norm_states['.'.join([name, k])] = v
    return async_norm_states


@HOOKS.register_module()
class SyncNormHook(Hook):
    """Synchronize Norm states after training epoch, currently used in YOLOX.

    Args:
        no_aug_epochs (int): The number of latter epochs in the end of the
            training to switch to synchronizing norm interval. Default: 15.
        interval (int): Synchronizing norm interval. Default: 1.
    """

    def __init__(self, no_aug_epochs=15, interval=1, **kwargs):
        super(SyncNormHook, self).__init__()
        self.interval = interval
        self.no_aug_epochs = no_aug_epochs

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        if (epoch + 1) == runner.max_epochs - self.no_aug_epochs:
            # Synchronize norm every epoch.
            self.interval = 1

    def after_train_epoch(self, runner):
        """Synchronizing norm."""
        epoch = runner.epoch
        module = runner.model
        if (epoch + 1) % self.interval == 0:
            _, world_size = get_dist_info()
            if world_size == 1:
                return
            norm_states = get_norm_states(module)
            norm_states = all_reduce_dict(norm_states, op='mean')
            module.load_state_dict(norm_states, strict=False)
