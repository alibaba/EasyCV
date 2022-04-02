# Copyright (c) Alibaba, Inc. and its affiliates.
import os

import torch
from mmcv.runner import Hook, get_dist_info

from .registry import HOOKS


@HOOKS.register_module
class SWAVHook(Hook):
    '''Hook in SWAV
    '''

    def __init__(self, gpu_batch_size=32, dump_path='data/', **kwargs):
        self.dump_path = dump_path
        self.queue_length = None
        self.rank, self.world_size = get_dist_info()
        self.batch_size = gpu_batch_size
        if not os.path.exists(self.dump_path):
            os.makedirs(self.dump_path)

    def before_run(self, runner):
        runner.model.module.queue = None
        runner.model.module.queue_path = os.path.join(
            self.dump_path, 'queue' + str(self.rank) + '.pth')

        if os.path.isfile(runner.model.module.queue_path):
            runner.model.module.queue = torch.load(
                runner.model.module.queue_path)['queue']
        # the queue needs to be divisible by the batch size
        # print(type(runner.model.module))

        self.queue_length = runner.model.module.config['queue_length']
        self.queue_length -= self.queue_length % (
            self.batch_size * self.world_size)

    def before_train_epoch(self, runner):
        if self.queue_length > 0 and runner.epoch >= runner.model.module.config[
                'epoch_queue_starts'] and runner.model.module.queue is None:
            runner.model.module.queue = torch.zeros(
                len(runner.model.module.config['crops_for_assign']),
                self.queue_length // self.world_size,
                runner.model.module.feat_dim,
            ).cuda()
        return

    def after_train_epoch(self, runner):
        if runner.model.module.queue is not None:
            torch.save({'queue': runner.model.module.queue},
                       runner.model.module.queue_path)
