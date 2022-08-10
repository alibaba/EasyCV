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
"""Pytorch extension for distributed training."""

from __future__ import absolute_import, division, print_function

from easycv.core.sailfish.activation import LogSoftmax  # noqa: F401
from easycv.core.sailfish.linear import ArcFaceLinear  # noqa: F401
from easycv.core.sailfish.linear import Linear  # noqa: F401
from easycv.core.sailfish.loss import ArcMarginLoss  # noqa: F401
from easycv.core.sailfish.loss import CrossEntropyLoss  # noqa: F401
from easycv.core.sailfish.loss import FocalLoss  # noqa: F401
from easycv.core.sailfish.loss import NLLLoss  # noqa: F401
from easycv.core.sailfish.loss import SoftmaxLoss  # noqa: F401
from easycv.core.sailfish.util import BiasUniformInitializer  # noqa: F401
from easycv.core.sailfish.util import DistributedParallel  # noqa: F401
from easycv.core.sailfish.util import KaimingUniformInitializer  # noqa: F401
from easycv.core.sailfish.util import ModelParallel  # noqa: F401
from easycv.core.sailfish.util import NormalInitializer  # noqa: F401
from easycv.core.sailfish.util import OnesInitializer  # noqa: F401
from easycv.core.sailfish.util import ParameterInitializer  # noqa: F401
from easycv.core.sailfish.util import RenormUniformInitializer  # noqa: F401
from easycv.core.sailfish.util import XavierUniformInitializer  # noqa: F401
from easycv.core.sailfish.util import ZerosInitializer  # noqa: F401
