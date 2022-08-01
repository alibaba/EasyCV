# Copyright (c) Alibaba, Inc. and its affiliates.
import logging
from distutils.version import LooseVersion

import torch
from mmcv.parallel import is_module_wrapper
from mmcv.runner import OptimizerHook as _OptimizerHook

from easycv.utils.dist_utils import get_dist_info
from easycv.utils.torchacc_util import is_torchacc_enabled

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

    def _get_module(self, runner):
        module = runner.model
        if is_module_wrapper(module):
            module = module.module
        return module

    def skip_ignore_key(self, runner):
        module = self._get_module(runner)
        for name, p in module.named_parameters():
            for k, epoch in zip(self.ignore_key, self.ignore_key_epoch):
                if k in name and runner.epoch < epoch:
                    p.grad = None

    def multiply_grad(self, runner):
        module = self._get_module(runner)
        for name, p in module.named_parameters():
            for k, ratio in zip(self.multiply_key, self.multiply_rate):
                if k in name:
                    p.grad = p.grad * ratio

    def adapt_torchacc(self, runner):
        import torchacc.torch_xla.core.xla_model as xm
        gradients = xm._fetch_gradients(runner.optimizer)
        xm.all_reduce(
            xm.REDUCE_SUM, gradients, scale=1.0 / xm.xrt_world_size())

    def after_train_iter(self, runner):
        if not torch.isnan(runner.outputs['loss']):
            runner.outputs['loss'] /= self.update_interval
            runner.outputs['loss'].backward()

            self.skip_ignore_key(runner)
            self.multiply_grad(runner)

            if self.every_n_iters(runner, self.update_interval):
                if is_torchacc_enabled():
                    self.adapt_torchacc(runner)

                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())

                runner.optimizer.step()
                runner.optimizer.zero_grad()
        else:
            rank, _ = get_dist_info()
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
                 ignore_key_epoch=[],
                 loss_scale={}):
        '''
            ignore_key: [str,...], ignore_key[i], name of parameters, which's gradient will be set to zero before every optimizer step when epoch < ignore_key_epoch[i]
            ignore_key_epoch: [int,...], epoch < ignore_key_epoch[i], ignore_key[i]'s gradient will be set to zero.
            loss_scale (float | dict): grade scale config. If loss_scale is a float, static loss scaling will be used with the specified scale.
                It can also be a dict containing arguments of GradScalar. For Pytorch >= 1.6, we use official torch.cuda.amp.GradScaler.
                please refer to: https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler for the parameters.
        '''
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.update_interval = update_interval
        self.ignore_key = ignore_key
        self.ignore_key_epoch = ignore_key_epoch
        self._scale_update_param = None

        if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
            if isinstance(loss_scale, float):
                self._scale_update_param = loss_scale
                self.scaler = amp.GradScaler(init_scale=loss_scale)
            elif isinstance(loss_scale, dict):
                self.scaler = amp.GradScaler(**loss_scale)
            else:
                raise ValueError(
                    '`loss_scale` type must be in [float, dict], but got {loss_scale}'
                )

    def before_run(self, runner):
        logging.info('open fp16')

        # set `fp16_enabled` flag. adapt to mmdet
        # TODO: find a more pretty way to adapt mmdet
        for m in runner.model.modules():
            if hasattr(m, 'fp16_enabled'):
                m.fp16_enabled = True

        runner.optimizer.zero_grad()

    def after_train_iter(self, runner):
        loss = runner.outputs['loss'] / self.update_interval

        if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
            self.scaler.scale(loss).backward()

            self.skip_ignore_key(runner)

            if self.every_n_iters(runner, self.update_interval):

                # gradients allreduce must before scaler.unscale_
                if is_torchacc_enabled():
                    self.adapt_torchacc(runner)

                self.scaler.unscale_(runner.optimizer)
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())

                self.scaler.step(runner.optimizer)
                self.scaler.update(self._scale_update_param)
                runner.optimizer.zero_grad()
        else:
            with amp.scale_loss(loss, runner.optimizer) as scaled_loss:
                scaled_loss.backward()

            self.skip_ignore_key(runner)

            if self.every_n_iters(runner, self.update_interval):
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())

                runner.optimizer.step()
                runner.optimizer.zero_grad()
