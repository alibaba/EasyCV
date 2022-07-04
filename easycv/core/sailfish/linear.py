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
"""Linear modules."""

from __future__ import absolute_import, division, print_function
import math

import torch

from easycv.core.sailfish.util import (BiasUniformInitializer,
                                       KaimingUniformInitializer,
                                       ModelParallel, RenormUniformInitializer)


class Linear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data.
  """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 weight_initializer=None,
                 bias_initializer=None,
                 parallel=None):
        super(Linear, self).__init__()
        if isinstance(parallel, ModelParallel):
            if out_features % parallel.world_size != 0:
                raise ValueError(
                    'out_features must be divided by parallel.world_size')
            self.out_features = out_features // parallel.world_size
        else:
            self.out_features = out_features
        self.in_features = in_features
        self.weight = torch.nn.Parameter(
            torch.Tensor(self.out_features, self.in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(self.out_features))
        else:
            self.register_parameter('bias', None)
        self.weight_initializer = weight_initializer
        if weight_initializer is None:
            self.weight_initializer = KaimingUniformInitializer(math.sqrt(5))
        self.bias_initializer = bias_initializer
        if bias_initializer is None:
            self.bias_initializer = BiasUniformInitializer(self.in_features)
        self.reset_parameters()
        self.parallel = parallel

    def reset_parameters(self):
        r"""Reset parameter."""
        self.weight_initializer(self.weight)
        if self.bias is not None:
            self.bias_initializer(self.bias)

    def forward(self, features):  # pylint: disable=arguments-differ
        features = features.type(dtype=self.weight.dtype)
        return torch.nn.functional.linear(features, self.weight, self.bias)


class ArcFaceLinear(torch.nn.Module):
    r"""Applies a ArcFace transformation to the incoming data.
      See https://arxiv.org/abs/1801.05599 .
  """

    def __init__(
            self,
            in_features,
            out_features,
            margin=0.5,
            scale=64.0,  # See normface https://arxiv.org/abs/1704.06369
            fast_phi=False,
            epsilon=0,
            weight_initializer=None,
            l2_norm=False,
            parallel=None):
        super(ArcFaceLinear, self).__init__()
        if isinstance(parallel, ModelParallel):
            if out_features % parallel.world_size != 0:
                raise ValueError(
                    'out_features must be divided by parallel.world_size')
            self.out_features = out_features // parallel.world_size
        else:
            self.out_features = out_features
        self.in_features = in_features
        self.margin = margin  # Angular margin penalty.
        self.scale = scale  # Radius of hybershpere.
        self.fast_phi = fast_phi
        self.epsilon = epsilon
        self.weight_initializer = weight_initializer
        if weight_initializer is None:
            self.weight_initializer = RenormUniformInitializer()
        self.l2_norm = l2_norm
        self.parallel = parallel

        self.weight = torch.nn.Parameter(
            torch.Tensor(self.out_features, self.in_features))
        self.reset_parameters()

        self._cos_margin = math.cos(margin)
        self._sin_margin = math.sin(margin)
        self._threshold = math.cos(math.pi - margin)
        self._min = math.sin(math.pi - margin) * self.margin

    def reset_parameters(self):
        r"""Reset parameters."""
        self.weight_initializer(self.weight)

    def forward(self, features, target):  # pylint: disable=arguments-differ
        r"""Compute ::math`\phi = \cos(\theta + margin)` and logits."""
        # (N, E) x (E, C) -> (N, C)
        features = features.type(dtype=self.weight.dtype)
        if self.l2_norm:
            features_norm = torch.norm(features, 2, 1, True)
            features = torch.div(features, features_norm)
            weight_norm = torch.norm(self.weight, 2, 0, True)
            weight = torch.div(self.weight, weight_norm)
        else:
            features = torch.nn.functional.normalize(features)
            weight = torch.nn.functional.normalize(self.weight)
        cosine = torch.nn.functional.linear(features, weight)
        cosine = cosine.clamp(-1, 1)  # for numerical stability
        sine = torch.sqrt(1. + self.epsilon - cosine * cosine)
        phi = cosine * self._cos_margin - sine * self._sin_margin
        phi = phi.type(dtype=cosine.dtype)
        if self.fast_phi:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self._threshold, phi,
                              cosine - self._min)
        if isinstance(self.parallel, ModelParallel):
            mask = self.parallel.correct_mask(target, cosine)
        else:
            mask = torch.zeros(
                cosine.size(), device=cosine.device, dtype=cosine.dtype)
            mask.scatter_(1, target.view(-1, 1).long(), 1)
        logits = (mask * phi) + ((1.0 - mask) * cosine)
        logits *= self.scale
        return logits
