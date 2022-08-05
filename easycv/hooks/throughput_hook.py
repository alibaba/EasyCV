# Copyright (c) Alibaba, Inc. and its affiliates.
import time

from mmcv.runner.hooks import Hook
from torch import distributed as dist

from easycv.hooks.registry import HOOKS


@HOOKS.register_module()
class ThroughputHook(Hook):
    """Count the throughput per second of all steps in the history.
    `warmup_iters` can be set to skip the calculation of the first few steps,
    if the initialization of the first few steps is slow.
    """

    def __init__(self, warmup_iters=0, **kwargs) -> None:
        self.warmup_iters = warmup_iters
        self._iter_count = 0
        self._start = False

    def _reset(self):
        self._start_time = time.time()
        self._iter_count = 0
        self._start = False

    def before_train_epoch(self, runner):
        """reset per epoch
        """
        self._reset()

    def before_train_iter(self, runner):
        if not self._start and self._iter_count == self.warmup_iters:
            self._start_time = time.time()
            self._start = True

    def after_train_iter(self, runner):
        self._iter_count += 1
        key = 'avg throughput'

        batch_size = runner.data_loader.batch_size
        num_gpus = dist.get_world_size()
        total_batch_size = batch_size * num_gpus

        # The LoggerHook will average the log_buffer of the latest interval,
        # but we want to use the total time to calculate the throughput,
        # so we delete the historical buffers of the key to ensure that
        # the value printed each time is the total historical average
        if key in runner.log_buffer.val_history:
            runner.log_buffer.val_history[key] = []
            runner.log_buffer.n_history[key] = []

        total_time = time.time() - self._start_time
        throughput = max(0,
                         (self._iter_count -
                          self.warmup_iters)) * total_batch_size / total_time
        runner.log_buffer.update({key: throughput}, count=self._iter_count)
