# Copyright (c) Alibaba, Inc. and its affiliates.
import time

from mmcv.runner import Hook, get_dist_info

from .registry import HOOKS


@HOOKS.register_module
class TIMEHook(Hook):
    '''
    This hook to show time for runner running process
    '''

    def __init__(self, end_momentum=1., **kwargs):
        self.end_infer = 0
        self.rank, self.num_replicas = get_dist_info()
        self.now_time = lambda: int(round(time.time() * 1000))

    def before_train_iter(self, runner):
        self.end_load = self.now_time()
        if self.rank == 0:
            print(self.rank,
                  ' load data need : %d ms' % (self.end_load - self.end_infer))

    def after_train_iter(self, runner):
        self.end_infer = self.now_time()
        if self.rank == 0:
            print(
                self.rank,
                ' infer model need : %d ms' % (self.end_infer - self.end_load))
