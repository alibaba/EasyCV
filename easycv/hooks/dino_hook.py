# Copyright (c) Alibaba, Inc. and its affiliates.
import time

import numpy as np
import torch
from mmcv.runner import Hook, get_dist_info

from .registry import HOOKS


def cosine_scheduler(base_value,
                     final_value,
                     epochs,
                     niter_per_ep,
                     warmup_epochs=0,
                     start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value,
                                      warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


@HOOKS.register_module
class DINOHook(Hook):
    '''Hook in DINO
    '''

    def __init__(self,
                 momentum_teacher=0.996,
                 weight_decay=0.04,
                 weight_decay_end=0.4,
                 **kwargs):
        self.momentum_teacher = momentum_teacher
        self.weight_decay = weight_decay
        self.weight_decay_end = weight_decay_end

    def before_run(self, runner):
        # call model init
        runner.model.module.init_before_train()

        try:
            self.rank, self.world_size = get_dist_info()
        except:
            self.rank = 0
            self.world_size = 1

        max_progress = runner.max_epochs
        self.epoch_length = runner.data_loader[0].__len__()
        self.momentum_schedule = cosine_scheduler(self.momentum_teacher, 1,
                                                  max_progress,
                                                  self.epoch_length)
        self.wd_schedule = cosine_scheduler(self.weight_decay,
                                            self.weight_decay_end,
                                            max_progress, self.epoch_length)
        self.optimizer = runner.optimizer
        runner.model.module.this_loss = 0
        runner.model.module.count = 0

        self.epoch_total_loss = 0
        self.count = 0

    def before_train_iter(self, runner):
        cur_iter = runner.iter

        # setup weight decay
        for i, param_group in enumerate(self.optimizer.param_groups):
            if i == 0:  # only the first group is regularized
                param_group['weight_decay'] = self.wd_schedule[cur_iter]

        # call model ema
        if cur_iter > 0:
            runner.model.module.momentum_update_key_encoder(
                self.momentum_schedule[cur_iter])

    def after_train_iter(self, runner):
        if self.world_size > 1:
            t = torch.tensor(
                [runner.model.module.count, runner.model.module.this_loss],
                dtype=torch.float64,
                device='cuda')
            torch.distributed.barrier()
            torch.distributed.all_reduce(t)
            t = t.tolist()
            self.count += int(t[0])
            self.epoch_total_loss += t[1]
        else:
            self.count += int(runner.model.module.count)
            self.epoch_total_loss += runner.model.module.this_loss

        if runner.iter % 10 == 0 and self.rank == 0:
            print(' wd : %.4f  momentum : %.4f  total_avg_loss : %.4f' %
                  (self.wd_schedule[runner.iter],
                   self.momentum_schedule[runner.iter],
                   self.epoch_total_loss / self.count))

    def before_train_epoch(self, runner):
        # reset epoch loss
        self.epoch_total_loss = 0
        self.count = 0
        torch.cuda.empty_cache()
        # Make sure `torch.cuda.empty_cache` is done and all cache is cleaned
        time.sleep(3)
        runner.model.module.cur_epoch = runner.epoch
