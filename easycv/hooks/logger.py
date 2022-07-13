# Copyright (c) Alibaba, Inc. and its affiliates.
import torch
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger import LoggerHook


@HOOKS.register_module()
class PreLoggerHook(LoggerHook):

    def fetch_tensor(self, runner, n=0):
        """Fetch latest n values or all values, process tensor type, convert to numpy for dump logs."""
        assert n >= 0
        for key in runner.log_buffer.val_history:
            values = runner.log_buffer.val_history[key][-n:]

            for i, v in enumerate(values):
                if isinstance(v, torch.Tensor):
                    values[i] = v.clone().detach().cpu().numpy()

            runner.log_buffer.val_history[key][-n:] = values

    def after_train_iter(self, runner):
        if self.by_epoch and self.every_n_inner_iters(runner, self.interval):
            self.fetch_tensor(runner, self.interval)
        elif not self.by_epoch and self.every_n_iters(runner, self.interval):
            self.fetch_tensor(runner, self.interval)
        elif self.end_of_epoch(runner) and not self.ignore_last:
            # not precise but more stable
            self.fetch_tensor(runner, self.interval)

    def after_val_epoch(self, runner):
        self.fetch_tensor(runner)
