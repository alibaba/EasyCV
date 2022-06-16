# Copyright 2019 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Functions."""

from __future__ import absolute_import, division, print_function

import torch


class _Cat(torch.autograd.Function):
    """Concat inputs."""

    @staticmethod
    def forward(ctx, inputs, dim, rank, world_size):  # noqa: E501 # pylint: disable=arguments-differ
        r"""Cat is defined as:
    .. math::
      \text{all_cat}(x_i) = \bigoplus_j x_j
    """
        ctx.dim = dim
        ctx.rank = rank
        ctx.world_size = world_size
        all_inputs = [
            torch.zeros(
                inputs.size(), dtype=inputs.dtype, device=inputs.device)
            for _ in range(world_size)
        ]
        torch.distributed.all_gather(all_inputs, inputs)
        output = torch.cat(all_inputs, dim=dim)
        output.requires_grad_()
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        r"""Gradient of Cat is defined as:
    .. math::
      \nabla \text{all_cat}(x_i) =  \text{split}(\nabla x_i)
    """
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input)
        grad_input_dim_size = grad_input.size()[ctx.dim]
        assert grad_input_dim_size % ctx.world_size == 0
        split_size = grad_input_dim_size // ctx.world_size
        grad_input_splits = torch.split(grad_input, split_size, dim=ctx.dim)
        return grad_input_splits[ctx.rank], None, None, None


def all_cat(inputs, dim=0, rank=0, world_size=1):
    return _Cat.apply(inputs, dim, rank, world_size)


class _Sum(torch.autograd.Function):
    """Sum inputs."""

    @staticmethod
    def forward(_, inputs):  # pylint: disable=arguments-differ
        r"""Sum is defined as:
    .. math::
      \text{all_sum}(x_i) = \sum_j x_j
    """
        inputs_sum = inputs.clone()
        torch.distributed.all_reduce(inputs_sum)
        inputs_sum.requires_grad_()
        return inputs_sum

    @staticmethod
    def backward(_, grad_output):  # pylint: disable=arguments-differ
        r"""Gradient of Sum is defined as:
    .. math::
      \nabla \text{all_sum}(x_i) = \sum_j\nabla x_j
    """
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input)
        return grad_input


def all_sum(inputs):
    return _Sum.apply(inputs)


class _LogSoftmax(torch.autograd.Function):
    """Compute log softmax of logits."""

    @staticmethod
    def forward(ctx, logits, epsilon):  # pylint: disable=arguments-differ
        r"""LogSoftmax is defined as:
    .. math::
      \log(\text{softmax}(x_i))
        = \log\left(\frac{\text{e}^{x_i}}{\sum_j\text{e}^{x_j}}\right)
        = x_i - \log\sum_j\text{e}^{x_j}

    For numerical stability, it subtracts the maximum value for every logits:
    .. math::
      \log(\text{softmax}(x_i))
        = \hat{x_i} - \log\sum_j\text{e}^{\hat{x_j}},
        \hat{x} = x - \max_j{x_j}
    """
        ctx.logits_dtype = logits.dtype
        logits_max = torch.max(logits, dim=1).values
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX)
        logits = logits - logits_max.view(-1, 1)
        logits_exp = torch.exp(logits)
        logits_exp_sum = torch.sum(logits_exp, dim=1)
        torch.distributed.all_reduce(logits_exp_sum)
        logits_exp_sum_log = torch.log(logits_exp_sum + epsilon)
        prob_log = logits - logits_exp_sum_log.view(-1, 1)
        ctx.save_for_backward(prob_log)
        prob_log.requires_grad_()
        return prob_log

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        r"""Gradient of LogSoftmax is defined as:
    .. math::
      \nabla\log(\text{softmax}(x_i))
        = \nabla x_i - \text{softmax}(x_i) \sum_j \nabla x_j
    """
        grad_output_sum = torch.sum(grad_output, dim=1)
        torch.distributed.all_reduce(grad_output_sum)
        prob_log, = ctx.saved_tensors
        grad_input = torch.exp(prob_log) * grad_output_sum.view(-1, 1)
        grad_input = grad_output - grad_input
        grad_input = grad_input.type(dtype=ctx.logits_dtype)
        return grad_input, None


def all_log_softmax(logits, epsilon=1e-8):
    return _LogSoftmax.apply(logits, epsilon)


class _NLLLoss(torch.autograd.Function):
    """calculate NLLLoss from mask."""

    @staticmethod
    def forward(ctx, inputs, correct_mask):  # pylint: disable=arguments-differ
        ctx.inputs_size = inputs.size()
        ctx.save_for_backward(correct_mask)
        loss = torch.sum(inputs * correct_mask) / -ctx.inputs_size[0]
        torch.distributed.all_reduce(loss)
        loss.requires_grad_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):  # pylint: disable=arguments-differ
        correct_mask, = ctx.saved_tensors
        grad_input = grad_output.repeat(*ctx.inputs_size)
        grad_input = grad_input * correct_mask / -ctx.inputs_size[0]
        return grad_input, None


def all_nll_loss(inputs, correct_mask):
    return _NLLLoss.apply(inputs, correct_mask)


def shard_target_and_mask(target, output_features, rank=0):
    target_shard_begin = output_features * rank
    target_shard_end = output_features * (rank + 1)
    target_shard_lmask = torch.ge(target, target_shard_begin)
    target_shard_rmask = torch.lt(target, target_shard_end)
    target_mask = (target_shard_lmask * target_shard_rmask).to(target.device)
    target_shard = (target - target_shard_begin) * target_mask.long()
    return target_shard, target_mask


def shard_correct_mask(target, inputs, rank=0):
    """Get correct mask of inputs."""
    inputs_size = inputs.size()
    target_shard_begin = inputs_size[1] * rank
    target_shard_end = inputs_size[1] * (rank + 1)
    target_shard_lmask = torch.ge(target, target_shard_begin)
    target_shard_rmask = torch.lt(target, target_shard_end)
    target_mask = (target_shard_lmask * target_shard_rmask).to(target.device)
    target_shard = (target - target_shard_begin) * target_mask.long()
    mask = torch.zeros(inputs_size, device=inputs.device, dtype=inputs.dtype)
    mask.scatter_(1, target_shard.view(-1, 1).long(), 1)
    mask.masked_fill_((~target_mask).view(-1, 1).expand(*inputs.size()), 0)
    return mask


def shard_correct_predictions(target, logits, world_size=1):
    r"""Calculate correct predictions for logits."""
    shard_max_logits, shard_expected_class = torch.max(logits, dim=1)
    all_max_logits = [
        torch.zeros(
            shard_max_logits.size(),
            dtype=shard_max_logits.dtype,
            device=shard_max_logits.device) for _ in range(world_size)
    ]
    torch.distributed.all_gather(all_max_logits, shard_max_logits)
    all_max_logits = torch.cat([t.view(-1, 1) for t in all_max_logits], dim=1)
    rank_pred = torch.max(all_max_logits, dim=1)[1].view(-1, 1)
    all_shard_pred = [
        torch.zeros(
            shard_expected_class.size(),
            dtype=shard_expected_class.dtype,
            device=shard_expected_class.device) for _ in range(world_size)
    ]
    torch.distributed.all_gather(all_shard_pred, shard_expected_class)
    all_shard_pred = torch.cat([t.view(-1, 1) for t in all_shard_pred], dim=1)
    all_shard_pred_mask = torch.zeros(
        all_shard_pred.size(),
        device=all_shard_pred.device,
        dtype=all_shard_pred.dtype)
    all_shard_pred_mask.scatter_(1, rank_pred.long(), 1)
    shard_pred = torch.sum(
        all_shard_pred * all_shard_pred_mask, dim=1).view(-1, 1)
    pred = shard_pred + rank_pred * logits.size()[1]
    return (pred == target.data.view_as(pred)).sum().item()


def shard_topk_correct_predictions(target, logits, k, world_size=1):
    r"""Calculate correct predictions for logits."""
    # Step 1: Compute top-k of shard logits.
    logits_topk, logits_topk_idx = torch.topk(logits, k, dim=1)
    all_logits_topk = [
        torch.zeros(
            logits_topk.size(),
            dtype=logits_topk.dtype,
            device=logits_topk.device) for _ in range(world_size)
    ]
    torch.distributed.all_gather(all_logits_topk, logits_topk)
    all_logits_topk = torch.cat([t.view(-1, k) for t in all_logits_topk],
                                dim=1)

    all_logits_topk_idx = [
        torch.zeros(
            logits_topk_idx.size(),
            dtype=logits_topk_idx.dtype,
            device=logits_topk_idx.device) for _ in range(world_size)
    ]
    torch.distributed.all_gather(all_logits_topk_idx, logits_topk_idx)
    all_logits_topk_idx = torch.cat(
        [t.view(-1, k) for t in all_logits_topk_idx], dim=1)

    # Step 2: Compute global top-k indices.
    _, all_logits_topk_topk_idx = torch.topk(all_logits_topk, k, dim=1)
    all_logits_topk_topk_idx = all_logits_topk_topk_idx.view(-1, k)
    all_logits_topk_mask = torch.zeros(
        all_logits_topk_idx.size(),
        device=all_logits_topk_idx.device,
        dtype=all_logits_topk_idx.dtype)
    all_logits_topk_mask.scatter_(1, all_logits_topk_topk_idx.long(), 1)
    batch_size, shard_num_classes = logits.size()
    all_logits_topk_base = torch.cat([
        torch.ones([batch_size, k],
                   device=all_logits_topk_idx.device,
                   dtype=all_logits_topk_idx.dtype) * p * shard_num_classes
        for p in range(world_size)
    ],
                                     dim=1)
    all_logits_topk_gidx = all_logits_topk_base + all_logits_topk_idx

    # Step 3: Compute predictions and check.
    pred = torch.masked_select(all_logits_topk_gidx,
                               all_logits_topk_mask.type(torch.bool)).view(
                                   batch_size, k)
    return (pred == target.view(-1, 1)).sum().item()
