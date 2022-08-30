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
"""Loss modules."""

from __future__ import absolute_import, division, print_function

import torch

from easycv.core.sailfish.activation import LogSoftmax
from easycv.core.sailfish.linear import ArcFaceLinear, Linear
from easycv.core.sailfish.util import ModelParallel


class NLLLoss(torch.nn.Module):
    r"""The negative log likelihood loss for log probabilities. It is
  useful to train a classification problem with `C` classes.

  The `input` given through a forward call is expected to contain
  log-probabilities of each class. `input` has to be a Tensor of size either
  :math:`(N, C)` or :math:`(N, C, d_1, d_2, ..., d_K)`
  with :math:`K \geq 1` for the `K`-dimensional case (described later).

  Obtaining log-probabilities in a neural network is easily achieved by
  adding a  `LogSoftmax` layer in the last layer of your network.
  You may use `CrossEntropyLoss` instead, if you prefer not to add an
  extra layer.

  The `target` that this loss expects should be a class index in the range
  :math:`[0, C-1]` where `C = number\_classes`.

  NLLLoss is defined as:
  .. math::
      \ell(x, y) = -\frac{1}{N}\sum_{n=1}^N L_{i}

  Args:
    num_classes: total number of classes.
    focal: whether to use FocalLoss implementation.
    focal_gamm: The focusing parameter of FocalLoss.
    rank: rank of current replica.
    world_size: size of replicas.

  Shape:
    - Input: :math:`(\frac{N}{P}, C)` where `C = number of classes`, or
      :math:`(N, C, d_1, d_2, ..., d_K)` with :math:`K \geq 1`
      in the case of `K`-dimensional loss.
    - Target: :math:`(\frac{N}{P}, 1)` where each value is
      :math:`0 \leq \text{targets}[i] \leq C-1`, or
      :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in the case of
      K-dimensional loss.
    - Output: scalar.
      If :attr:`reduction` is ``'none'``, then the same size as the target:
      :math:`(N)`, or :math:`(N, d_1, d_2, ..., d_K)` with :math:`K \geq 1` in
      the case of K-dimensional loss.
  Examples::
    >>> m = LogSoftmax(...)
    >>> loss = NLLLoss(...)
    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = loss(m(input), target)
    >>> output.backward()
  """

    def __init__(self, focal=False, focal_gamma=0, parallel=None):
        super(NLLLoss, self).__init__()
        self.focal = focal
        self.focal_gamma = focal_gamma
        self.parallel = parallel

    def forward(self, logprob, target):  # pylint: disable=arguments-differ
        """Compute negative log likelihood loss from log-probs and target."""
        if isinstance(self.parallel, ModelParallel):
            with torch.no_grad():
                mask = self.parallel.correct_mask(target, logprob)
            loss = self.parallel.nll_loss(logprob, mask)
            if self.focal:
                loss_exp = torch.exp(-loss)
                loss = (1 - loss_exp)**self.focal_gamma * loss
            return loss
        loss = torch.nn.functional.nll_loss(logprob, target)
        if self.focal:
            loss_exp = torch.exp(-loss)
            loss = (1 - loss_exp)**self.focal_gamma * loss
        return loss


class CrossEntropyLoss(torch.nn.Module):
    r"""This criterion combines :func:`LogSoftmax` and
    :func:`NLLLoss` in one single class.
  """

    def __init__(self, epsilon=0, parallel=None):
        super(CrossEntropyLoss, self).__init__()
        self._log_softmax = LogSoftmax(epsilon=epsilon, parallel=parallel)
        self._nll_loss = NLLLoss(parallel=parallel)

    def forward(self, logits, target):  # pylint: disable=arguments-differ
        # 1. Compute log-probabilities for current shard:
        logprob = self._log_softmax(logits)

        # 2. Compute NLL loss for all shards.
        return self._nll_loss(logprob, target)


class FocalLoss(torch.nn.Module):
    r"""This criterion combines :func:`LogSoftmax` and
    :func:`NLLLoss` in one single class.
  """

    def __init__(self, gamma=0, epsilon=1e-8, parallel=None):
        super(FocalLoss, self).__init__()
        self._log_softmax = LogSoftmax(epsilon=epsilon, parallel=parallel)
        self._nll_loss = NLLLoss(
            focal=True, focal_gamma=gamma, parallel=parallel)

    def forward(self, logits, target):  # pylint: disable=arguments-differ
        logprob = self._log_softmax(logits)
        return self._nll_loss(logprob, target)


class SoftmaxLoss(torch.nn.Module):
    r"""This criterion combines :func:`Linear` and
    :func:`CrossEntropyLoss` in one single class.
  """

    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 epsilon=0,
                 weight_initializer=None,
                 bias_initializer=None,
                 parallel=None):
        super(SoftmaxLoss, self).__init__()
        self._linear = Linear(
            in_features,
            out_features,
            bias=bias,
            weight_initializer=weight_initializer,
            bias_initializer=bias_initializer,
            parallel=parallel)
        self._log_softmax = LogSoftmax(epsilon=epsilon, parallel=parallel)
        self._nll_loss = NLLLoss(parallel=parallel)
        self._parallel = parallel

    def forward(self, features, target):  # pylint: disable=arguments-differ
        if isinstance(self._parallel, ModelParallel):
            features = self._parallel.gather(features)
        logits = self._linear(features.squeeze())
        logprob = self._log_softmax(logits)
        if isinstance(self._parallel, ModelParallel):
            target = self._parallel.gather_target(target)
        return self._nll_loss(logprob, target)


class ArcMarginLoss(torch.nn.Module):
    r"""This criterion combines :func:`ArcFaceLinear` and
    :func:`CrossEntropyLoss` in one single class.
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
        super(ArcMarginLoss, self).__init__()
        self._linear = ArcFaceLinear(
            in_features,
            out_features,
            margin=margin,
            scale=scale,
            l2_norm=l2_norm,
            fast_phi=fast_phi,
            epsilon=epsilon,
            weight_initializer=weight_initializer,
            parallel=parallel)
        self._log_softmax = LogSoftmax(epsilon=epsilon, parallel=parallel)
        self._nll_loss = NLLLoss(parallel=parallel)
        self._parallel = parallel

    def forward(self, features, target):  # pylint: disable=arguments-differ
        if isinstance(self._parallel, ModelParallel):
            features = self._parallel.gather(features)
            target = self._parallel.gather_target(target)
        logits = self._linear(features.squeeze(), target)
        logprob = self._log_softmax(logits)
        return self._nll_loss(logprob, target)
