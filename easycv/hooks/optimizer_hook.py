# Copyright (c) Alibaba, Inc. and its affiliates.
from distutils.version import LooseVersion

import torch
from mmcv.runner import OptimizerHook as _OptimizerHook
from mmcv.utils import TORCH_VERSION, _BatchNorm, digit_version

from easycv.utils.dist_utils import get_dist_info
from easycv.utils.fp16_utils import wrap_fp16_model

if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
    from torch.cuda import amp
else:
    try:
        from apex import amp
    except ImportError:
        print(
            'Warning: apex not installed, please install apex from https://www.github.com/nvidia/apex if you want to use fp16.'
        )
        pass


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
    """FP16 optimizer hook (using PyTorch's implementation).

    If you are using PyTorch >= 1.6, torch.cuda.amp is used as the backend,
    to take care of the optimization procedure.

    Args:
        loss_scale (float | str | dict): Scale factor configuration.
            If loss_scale is a float, static loss scaling will be used with
            the specified scale. If loss_scale is a string, it must be
            'dynamic', then dynamic loss scaling will be used.
            It can also be a dict containing arguments of GradScalar.
            Defaults to 512. For Pytorch >= 1.6, mmcv uses official
            implementation of GradScaler. If you use a dict version of
            loss_scale to create GradScaler, please refer to:
            https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler
            for the parameters.

    Examples:
        >>> loss_scale = dict(
        ...     init_scale=65536.0,
        ...     growth_factor=2.0,
        ...     backoff_factor=0.5,
        ...     growth_interval=2000
        ... )
        >>> optimizer_hook = Fp16OptimizerHook(loss_scale=loss_scale)
    """
    def __init__(self,
                 update_interval=1,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 loss_scale=512.,
                 ignore_key=[],
                 ignore_key_epoch=[],
                 multiply_key=[],
                 multiply_rate=[]):
        '''
            ignore_key: [str,...], ignore_key[i], name of parameters, which's gradient will be set to zero before every optimizer step when epoch < ignore_key_epoch[i]
            ignore_key_epoch: [int,...], epoch < ignore_key_epoch[i], ignore_key[i]'s gradient will be set to zero.
        '''
        self.grad_clip = grad_clip
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.loss_scale = loss_scale
        self.ignore_key = ignore_key
        self.ignore_key_epoch = ignore_key_epoch
        self._scale_update_param = None

        # set update_interval
        self.update_interval = update_interval
        self.divisible_iters = 0
        self.remainder_iters = 0
        self.initialized = False

        if self.loss_scale == 'dynamic':
            self.loss_scaler = amp.GradScaler()
        elif isinstance(self.loss_scale, float):
            self._scale_update_param = self.loss_scale
            self.loss_scaler = amp.GradScaler(init_scale=self.loss_scale)
        elif isinstance(self.loss_scale, dict):
            self.loss_scaler = amp.GradScaler(**self.loss_scale)
        else:
            raise ValueError('loss_scale must be of type float, dict, or '
                                f'"dynamic", got {loss_scale}')
        # if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
        #     self.loss_scaler = amp.GradScaler()

    def has_batch_norm(self, module):
        if isinstance(module, _BatchNorm):
            return True
        for m in module.children():
            if self.has_batch_norm(m):
                return True
        return False

    def _init(self, runner):
        if runner.iter % self.update_interval != 0:
            runner.logger.warning(
                'Resume iter number is not divisible by update_interval in '
                'GradientCumulativeOptimizerHook, which means the gradient of '
                'some iters is lost and the result may be influenced slightly.'
            )

        if self.has_batch_norm(runner.model) and self.update_interval > 1:
            runner.logger.warning(
                'GradientCumulativeOptimizerHook may slightly decrease '
                'performance if the model has BatchNorm layers.')

        residual_iters = runner.max_iters - runner.iter

        self.divisible_iters = (
            residual_iters // self.update_interval * self.update_interval)
        self.remainder_iters = residual_iters - self.divisible_iters

        self.initialized = True

    def before_run(self, runner):
        """Preparing steps before Mixed Precision Training."""
        #runner.fp16_enable = True
        # wrap model mode to fp16
        wrap_fp16_model(runner.model)
        # resume from state dict
        if 'fp16' in runner.meta and 'loss_scaler' in runner.meta['fp16']:
            scaler_state_dict = runner.meta['fp16']['loss_scaler']
            self.loss_scaler.load_state_dict(scaler_state_dict)
        print('open fp16')
        #runner.optimizer.zero_grad()

    def copy_grads_to_fp32(self, fp16_net, fp32_weights):
        """Copy gradients from fp16 model to fp32 weight copy."""
        for fp32_param, fp16_param in zip(fp32_weights,
                                            fp16_net.parameters()):
            if fp16_param.grad is not None:
                if fp32_param.grad is None:
                    fp32_param.grad = fp32_param.data.new(
                        fp32_param.size())
                fp32_param.grad.copy_(fp16_param.grad)

    def copy_params_to_fp16(self, fp16_net, fp32_weights):
        """Copy updated params from fp32 weight copy to fp16 model."""
        for fp16_param, fp32_param in zip(fp16_net.parameters(),
                                            fp32_weights):
            fp16_param.data.copy_(fp32_param.data)

    def after_train_iter(self, runner):
        """Backward optimization steps for Mixed Precision Training. For
        dynamic loss scaling, please refer to
        https://pytorch.org/docs/stable/amp.html#torch.cuda.amp.GradScaler.

        1. Scale the loss by a scale factor.
        2. Backward the loss to obtain the gradients.
        3. Unscale the optimizer’s gradient tensors.
        4. Call optimizer.step() and update scale factor.
        5. Save loss_scaler state_dict for resume purpose.
        """
        if not self.initialized:
            self._init(runner)

        if runner.iter < self.divisible_iters:
            loss_factor = self.divisible_iters
        else:
            loss_factor = self.remainder_iters

        loss = runner.outputs['loss']
        loss = loss / loss_factor
        _, world_size = get_dist_info()

        if LooseVersion(torch.__version__) >= LooseVersion('1.6.0'):
            self.loss_scaler.scale(loss).backward()
            for name, p in runner.model.module.named_parameters():
                for k, epoch in zip(self.ignore_key, self.ignore_key_epoch):
                    if k in name and runner.epoch < epoch:
                        p.grad = None

            if (self.every_n_iters(runner, self.update_interval) or self.is_last_iter(runner)):
                self.loss_scaler.unscale_(runner.optimizer)
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())
                    
                self.loss_scaler.step(runner.optimizer)
                self.loss_scaler.update(self._scale_update_param)

                # save state_dict of loss_scaler
                runner.meta.setdefault(
                    'fp16', {})['loss_scaler'] = self.loss_scaler.state_dict()

                # clear grads
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
