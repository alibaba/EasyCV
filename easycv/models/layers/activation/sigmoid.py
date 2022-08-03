#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Tuple

from torch import Tensor, nn

from . import register_act_fn


@register_act_fn(name='sigmoid')
class Sigmoid(nn.Sigmoid):
    """
    Applies the sigmoid function
    """

    def __init__(self):
        super().__init__()

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        return input, 0.0, 0.0
