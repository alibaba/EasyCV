#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/detail/KernelUtils.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/AccumulateType.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/ConvUtils.h>

using namespace at;
std::tuple<Tensor, Tensor, Tensor> conv_depthwise3d_backward_cuda(
    const Tensor& grad_output,
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    const std::array<bool, 3> output_mask);

Tensor conv_depthwise3d_cuda(
    const Tensor& input,
    const Tensor& weight,
    IntArrayRef kernel_size,
    const Tensor& bias,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("conv_depthwise3d_backward_cuda", &conv_depthwise3d_backward_cuda,
        "conv_depthwise3d_backward_cuda");
  m.def("conv_depthwise3d_cuda", &conv_depthwise3d_cuda,
        "conv_depthwise3d_cuda");
}
