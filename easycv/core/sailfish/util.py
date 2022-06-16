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
"""Utility modules."""

from __future__ import absolute_import, division, print_function
import math

import torch

from easycv.core.sailfish.function import (all_cat, all_log_softmax,
                                           all_nll_loss, all_sum,
                                           shard_correct_mask,
                                           shard_correct_predictions,
                                           shard_target_and_mask,
                                           shard_topk_correct_predictions)


class DistributedParallel:
    """Base class of parallelism."""

    def __init__(self, rank, world_size):
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def correct_mask(self, target, inputs):
        mask = torch.zeros(
            inputs.size(), device=inputs.device, dtype=inputs.dtype)
        mask.scatter_(1, target.view(-1, 1).long(), 1)
        return mask

    def correct_predictions(self, target, logits, k=1):
        if k == 1:
            pred = torch.max(logits, dim=1)[1]
            return (pred == target.view(-1, 1)).sum().item()
        pred = torch.topk(logits, k, dim=1)[1]
        return (pred == target.view(-1, 1)).sum().item()

    def xavier_uniform_(self, weight, gain=1.):
        return torch.nn.init.xavier_uniform_(weight, gain=gain)


class ModelParallel(DistributedParallel):
    """All-to-All Model Parallelism."""

    def gather(self, inputs, dim=0, requires_grad=True):
        if requires_grad:
            return all_cat(
                inputs, dim=dim, rank=self.rank, world_size=self.world_size)
        all_inputs = [
            torch.zeros(
                inputs.size(), dtype=inputs.dtype, device=inputs.device)
            for _ in range(self.world_size)
        ]
        torch.distributed.all_gather(all_inputs, inputs)
        return torch.cat(all_inputs, dim=dim)

    def gather_target(self, target):
        return self.gather(target, requires_grad=False)

    def reduce_sum(self, inputs):
        return all_sum(inputs)

    def log_softmax(self, logits, epsilon=1e-8):
        return all_log_softmax(logits, epsilon=epsilon)

    def nll_loss(self, inputs, correct_mask):
        return all_nll_loss(inputs, correct_mask)

    def correct_mask(self, target, inputs):
        return shard_correct_mask(target, inputs, rank=self.rank)

    def target_and_mask(self, target, output_features):
        return shard_target_and_mask(target, output_features, rank=self.rank)

    def correct_predictions(self, target, logits, k=1):
        if k == 1:
            return shard_correct_predictions(
                target, logits, world_size=self.world_size)
        return shard_topk_correct_predictions(
            target, logits, k, world_size=self.world_size)


class ParameterInitializer:
    r"""Base class for parameter initializer."""

    def __call__(self, param, shard_rank=0, num_shards=1):
        raise NotImplementedError


class ZerosInitializer(ParameterInitializer):

    def __call__(self, param, parallel=None):
        torch.nn.init.zeros_(param)


class OnesInitializer(ParameterInitializer):

    def __call__(self, param, parallel=None):
        torch.nn.init.ones_(param)


class XavierUniformInitializer(ParameterInitializer):

    def __init__(self, gain=1.):
        self.gain = gain

    def __call__(self, param, parallel=None):
        if isinstance(parallel, ModelParallel):
            if param.dim() != 2:
                raise ValueError(
                    'param with dimensions other than 2 not supported')
            r = param.size(1) + param.size(0) * parallel.world_size
            a = self.gain * math.sqrt(3.0) * math.sqrt(2.0 / float(r))
            torch.nn.init.uniform_(param, -a, a)
        else:
            torch.nn.init.xavier_uniform_(param, gain=self.gain)


class KaimingUniformInitializer(ParameterInitializer):

    def __init__(self, bound):
        self.bound = bound

    def __call__(self, param, parallel=None):
        torch.nn.init.kaiming_uniform_(param, a=self.bound)


class BiasUniformInitializer(ParameterInitializer):

    def __init__(self, weight_in_features):
        self.weight_in_features = weight_in_features

    def __call__(self, param, parallel=None):
        a = 1 / math.sqrt(float(self.weight_in_features))
        torch.nn.init.uniform_(param, -a, a)


class RenormUniformInitializer(ParameterInitializer):

    def __init__(self, maxnorm=1e-5, scale=1e5):
        self.maxnorm = maxnorm
        self.scale = scale

    def __call__(self, param, parallel=None):
        param.data.uniform_(-1, 1).renorm_(
            2, 0, maxnorm=self.maxnorm).mul_(self.scale)


class NormalInitializer(ParameterInitializer):

    def __call__(self, param, parallel=None):
        torch.nn.init.normal_(param)
