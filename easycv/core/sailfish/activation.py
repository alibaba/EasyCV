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
"""Activation modules."""

from __future__ import absolute_import, division, print_function

import torch

from easycv.core.sailfish.util import ModelParallel


class LogSoftmax(torch.nn.Module):
    r"""Applies the :math:`\log(\text{Softmax}(x))` function to an n-dimensional
  input Tensor rescaling them so that the elements of the
  n-dimensional output Tensor lie in the range (0, 1].

  Shape:
    - Input: :math:`(*)` where `*` means, any number of additional
      dimensions
    - Output: :math:`(*)`, same shape as the input

  Returns:
    a Tensor of the same dimension and shape as the input with
    values in the range (-inf,0].

  Examples::
    >>> m = LogSoftmax()
    >>> input = torch.randn(2, 3)
    >>> output = m(input)
  """

    def __init__(self, epsilon=0, parallel=None):
        super(LogSoftmax, self).__init__()
        self.epsilon = epsilon
        self.parallel = parallel

    def forward(self, logits):  # pylint: disable=arguments-differ
        if isinstance(self.parallel, ModelParallel):
            return self.parallel.log_softmax(logits, epsilon=self.epsilon)
        return torch.nn.functional.log_softmax(logits, _stacklevel=5)
