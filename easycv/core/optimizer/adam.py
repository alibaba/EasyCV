# Copyright (c) Alibaba, Inc. and its affiliates.
import math
from distutils.version import LooseVersion
from typing import List

import torch
from mmcv.runner.optimizer.builder import OPTIMIZERS
from torch import Tensor
from torch.optim import AdamW as _AdamW


def adamw(params: List[Tensor], grads: List[Tensor], exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor], max_exp_avg_sqs: List[Tensor],
          state_steps: List[int], amsgrad: bool, beta1: float, beta2: float,
          lr: float, weight_decay: float, eps: float):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(
                max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() /
                     math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)


if LooseVersion(torch.__version__) <= LooseVersion('1.9.0'):

    @OPTIMIZERS.register_module(force=True)
    class AdamW(_AdamW):
        """
        torch1.8 bug UnboundLocalError: local variable 'beta1' referenced before assignment
        bugfix reference: https://github.com/pytorch/pytorch/issues/55740
        """

        @torch.no_grad()
        def step(self, closure=None):
            """Performs a single optimization step.
            Args:
                closure (callable, optional): A closure that reevaluates the model
                    and returns the loss.
            """
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                state_sums = []
                max_exp_avg_sqs = []
                state_steps = []
                amsgrad = group['amsgrad']
                beta1, beta2 = group['betas']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'AdamW does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]

                    # State initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format)
                        if amsgrad:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(
                                p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if amsgrad:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

                adamw(params_with_grad, grads, exp_avgs, exp_avg_sqs,
                      max_exp_avg_sqs, state_steps, amsgrad, beta1, beta2,
                      group['lr'], group['weight_decay'], group['eps'])

            return loss
