#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from typing import Optional, Tuple

from torch import Tensor, nn

from . import register_norm_fn


@register_norm_fn(name='batch_norm')
@register_norm_fn(name='batch_norm_2d')
class BatchNorm2d(nn.BatchNorm2d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 4D input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, H, W)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input channels,
        :math:`H` is the input height, and :math:`W` is the input width
        - Output: same shape as the input
    """

    def __init__(self,
                 num_features: int,
                 eps: Optional[float] = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: Optional[bool] = True,
                 track_running_stats: Optional[bool] = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


@register_norm_fn(name='batch_norm_1d')
class BatchNorm1d(nn.BatchNorm1d):
    """
    Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 2D or 3D input tensor

    Args:
        num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C)` or :math:`(N, C, L)`
        eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
        momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
        affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
        track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

    Shape:
        - Input: :math:`(N, C)` or :math:`(N, C, L)` where :math:`N` is the batch size,
        :math:`C` is the number of input channels,  and :math:`L` is the sequence length
        - Output: same shape as the input
    """

    def __init__(self,
                 num_features: int,
                 eps: Optional[float] = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: Optional[bool] = True,
                 track_running_stats: Optional[bool] = True,
                 *args,
                 **kwargs) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0


@register_norm_fn(name='batch_norm_3d')
class BatchNorm3d(nn.BatchNorm3d):

    def __init__(self,
                 num_features: int,
                 eps: Optional[float] = 1e-5,
                 momentum: Optional[float] = 0.1,
                 affine: Optional[bool] = True,
                 track_running_stats: Optional[bool] = True,
                 *args,
                 **kwargs) -> None:
        """
        Applies a `Batch Normalization <https://arxiv.org/abs/1502.03167>`_ over a 5D input tensor

        Args:
            num_features (Optional, int): :math:`C` from an expected input of size :math:`(N, C, D, H, W)`
            eps (Optional, float): Value added to the denominator for numerical stability. Default: 1e-5
            momentum (Optional, float): Value used for the running_mean and running_var computation. Default: 0.1
            affine (bool): If ``True``, use learnable affine parameters. Default: ``True``
            track_running_stats: If ``True``, tracks running mean and variance. Default: ``True``

        Shape:
            - Input: :math:`(N, C, D, H, W)` where :math:`N` is the batch size, :math:`C` is the number of input
            channels, :math:`D` is the input depth, :math:`H` is the input height, and :math:`W` is the input width
            - Output: same shape as the input
        """
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def profile_module(self, input: Tensor) -> Tuple[Tensor, float, float]:
        # Since normalization layers can be fused, we do not count their operations
        params = sum([p.numel() for p in self.parameters()])
        return input, params, 0.0
