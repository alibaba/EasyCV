# Accelerated 3D Depthwise Convolution

This is seperate repo of my pull request (Accelerated 3D Depthwise Convolution), which is part of Pytorch 1.9.
This repo aim to support other people want to use the module without upgrade to latest cudnn or pytorch.

## Installation

prerequisite:

- Pytorch >= 1.6
- Python3

``` bash
python setup.py install
```

## Usage

```python

import torch
from depthwise_conv3d import DepthwiseConv3d

dtype = torch.float
conv = DepthwiseConv3d(2, 2, kernel_size=3, groups=2).to("cuda", dtype)
input = torch.randn(2, 2, 6, 6, 6, device="cuda", dtype=dtype).div_(2).requires_grad_()
output = conv(input)

```