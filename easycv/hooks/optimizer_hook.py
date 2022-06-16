# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from distutils.version import LooseVersion

import torch
from mmcv.runner import OptimizerHook as _OptimizerHook
from mmcv.runner.fp16_utils import wrap_fp16_model

import easycv.distributed as dist

if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
    from torch.cuda import amp
else:
    try:
        from apex import amp
    except ImportError:
        logging.warning(
            'apex not installed, please install apex from https://www.github.com/nvidia/apex if you want to use fp16.'
        )


class OptimizerHook(_OptimizerHook):

    def __init__(self,
                 update_interval=1,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 ignore_key=[],
                 ignore_key_epoch=[],
                 multiply_key=[],
                 multiply_rate=[]):
        '''
            ignore_key: [str,...], ignore_key[i], name of parameters, which's gradient will be set to zero before every optimizer step when epoch < ignore_key_epoch[i]
            ignore_key_epoch: [int,...], epoch < ignore_key_epoch[i], ignore_key[i]'s gradient will be set to zero.

            multiply_key:[str,...] multiply_key[i], name of parameters, which will set different learning rate ratio by multipy_rate
            multiply_rate:[float,...] multiply_rate[i], different ratio

        '''
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.ignore_key = ignore_key
        self.ignore_key_epoch = ignore_key_epoch
        self.multiply_key = multiply_key
        self.multiply_rate = multiply_rate

    def before_run(self, runner):
        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        if not torch.isnan(runner.outputs['loss']):
            runner.outputs['loss'] /= self.update_interval
            runner.outputs['loss'].backward()

            for name, p in runner.model.module.named_parameters():
                for k, epoch in zip(self.ignore_key, self.ignore_key_epoch):
                    if k in name and runner.epoch < epoch:
                        p.grad = None

            for name, p in runner.model.module.named_parameters():
                for k, ratio in zip(self.multiply_key, self.multiply_rate):
                    if k in name:
                        p.grad = p.grad * ratio

            if self.every_n_iters(runner, self.update_interval):
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())
                runner.optimizer.step()
                runner.optimizer.zero_grad()
        else:
            rank = dist.get_rank()
            # catch nan loss, not update, zero_grad to pass
            if rank == 0:
                runner.logger.info('catch nan loss in iter %d, epoch %d' %
                                   (runner.iter, runner.epoch))

            if self.every_n_iters(runner, self.update_interval):
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())
                runner.optimizer.zero_grad()


class AMPFP16OptimizerHook(OptimizerHook):

    def __init__(self,
                 update_interval=1,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 ignore_key=[],
                 ignore_key_epoch=[]):
        '''
            ignore_key: [str,...], ignore_key[i], name of parameters, which's gradient will be set to zero before every optimizer step when epoch < ignore_key_epoch[i]
            ignore_key_epoch: [int,...], epoch < ignore_key_epoch[i], ignore_key[i]'s gradient will be set to zero.
        '''
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.ignore_key = ignore_key
        self.ignore_key_epoch = ignore_key_epoch
        if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
            self.scaler = amp.GradScaler()
            self._scale_update_param = 512.

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training."""
        # wrap model mode to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if 'fp16' in runner.meta and 'scaler' in runner.meta['fp16']:
            scaler_state_dict = runner.meta['fp16']['scaler']
            self.scaler.load_state_dict(scaler_state_dict)
        print('open fp16')

    def after_train_iter(self, runner):
        loss = runner.outputs['loss'] / self.update_interval

        if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
            self.scaler.scale(loss).backward()
            for name, p in runner.model.module.named_parameters():
                for k, epoch in zip(self.ignore_key, self.ignore_key_epoch):
                    if k in name and runner.epoch < epoch:
                        p.grad = None

            if self.every_n_iters(runner, self.update_interval):
                self.scaler.unscale_(runner.optimizer)
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())
                self.scaler.step(runner.optimizer)
                self.scaler.update(self._scale_update_param)

                # save state_dict of scaler
                runner.meta.setdefault(
                    'fp16', {})['scaler'] = self.scaler.state_dict()

                runner.model.zero_grad()
                runner.optimizer.zero_grad()
        else:
            with amp.scale_loss(loss, runner.optimizer) as scaled_loss:
                scaled_loss.backward()

            for name, p in runner.model.module.named_parameters():
                for k, epoch in zip(self.ignore_key, self.ignore_key_epoch):
                    if k in name and runner.epoch < epoch:
                        p.grad = None

            if self.every_n_iters(runner, self.update_interval):
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())

                runner.optimizer.step()
                runner.optimizer.zero_grad()
