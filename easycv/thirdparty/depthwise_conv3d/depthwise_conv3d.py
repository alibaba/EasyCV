import math

import DWCONV_CUDA
import torch
from torch import nn
from torch.cuda.amp import custom_bwd, custom_fwd
from torch.nn.modules.utils import _triple


class DepthwiseConv3dFunction(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1,
                groups=1):
        ctx.stride = _triple(stride)
        ctx.padding = _triple(padding)
        ctx.dilation = _triple(dilation)
        ctx.kernel_size = _triple(weight.shape[2])
        ctx.groups = groups
        ctx.with_bias = bias is not None
        if not ctx.with_bias:
            bias = input.new_empty(0)  # fake tensor
        if not input.is_cuda:
            raise NotImplementedError
        if weight.requires_grad or input.requires_grad:
            ctx.save_for_backward(input, weight, bias)
        weight = weight.to(input.dtype)
        bias = bias.to(input.dtype)
        output = DWCONV_CUDA.conv_depthwise3d_cuda(
            input, weight, ctx.kernel_size, bias,
            ctx.stride,
            ctx.padding,
            ctx.dilation)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        if not grad_output.is_cuda:
            raise NotImplementedError
        input, weight, bias = ctx.saved_tensors
        grad_input = torch.zeros_like(input)
        grad_weight = torch.zeros_like(weight)
        grad_input, grad_weight, grad_bias = DWCONV_CUDA.conv_depthwise3d_backward_cuda(
            grad_output, grad_input, grad_weight,
            ctx.kernel_size,
            ctx.stride,
            ctx.padding,
            ctx.dilation, (True, True, True))
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None


depthwise_conv3d = DepthwiseConv3dFunction.apply


class DepthwiseConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(DepthwiseConv3d, self).__init__()
        assert in_channels % groups == 0, \
            'in_channels {} cannot be divisible by groups {}'.format(
                in_channels, groups)
        assert out_channels % groups == 0, \
            'out_channels {} cannot be divisible by groups {}'.format(
                out_channels, groups)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups

        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // self.groups, *self.kernel_size))
        self.with_bias = bias
        if self.with_bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.with_bias:
            self.bias.data.fill_(0)
    def forward(self, x):
        return depthwise_conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation,
                                self.groups, )
