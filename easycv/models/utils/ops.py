# Copyright (c) OpenMMLab. All rights reserved.
# Adapt from: https://github.com/open-mmlab/mmpose/blob/master/mmpose/models/utils/ops.py
import warnings

import torch
import torch.nn.functional as F


def resize_tensor(input,
                  size=None,
                  scale_factor=None,
                  mode='nearest',
                  align_corners=None,
                  warning=True):
    """Resize tensor with F.interpolate.

    Args:
        input (Tensor): the input tensor.
        size (Tuple[int, int]): output spatial size.
        scale_factor (float or Tuple[float]): multiplier for spatial size.
            If scale_factor is a tuple, its length has to match input.dim().
        mode (str): algorithm used for upsampling:
            'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'. Default: 'nearest'
        align_corners (bool): Geometrically, we consider the pixels of the input and output as squares rather than points.
            If set to True, the input and output tensors are aligned by the center points of their corner pixels,
            preserving the values at the corner pixels.

            If set to False, the input and output tensors are aligned by the corner points of their corner pixels,
            and the interpolation uses edge value padding for out-of-boundary values,
            making this operation independent of input size when scale_factor is kept the same.
            This only has an effect when mode is 'linear', 'bilinear', 'bicubic' or 'trilinear'.
    """
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > output_h:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    return F.interpolate(input, size, scale_factor, mode, align_corners)
