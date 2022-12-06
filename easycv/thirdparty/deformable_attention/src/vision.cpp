/*!
**************************************************************************************************
* Deformable DETR
* Copyright (c) 2020 SenseTime. All Rights Reserved.
* Licensed under the Apache License, Version 2.0 [see LICENSE for details]
**************************************************************************************************
* Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
**************************************************************************************************
*/

/*!
* Copyright (c) Facebook, Inc. and its affiliates.
* Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR
*/

#include "ms_deform_attn.h"
#include <torch/script.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ms_deform_attn_forward", &ms_deform_attn_forward, "ms_deform_attn_forward");
  m.def("ms_deform_attn_backward", &ms_deform_attn_backward, "ms_deform_attn_backward");
}

inline at::Tensor ms_deform_attn(
    const at::Tensor &value, 
    const at::Tensor &spatial_shapes,
    const at::Tensor &level_start_index,
    const at::Tensor &sampling_loc,
    const at::Tensor &attn_weight,
    const int64_t im2col_step) {
  return ms_deform_attn_forward(value, spatial_shapes, level_start_index, 
    sampling_loc, attn_weight, im2col_step);
}
static auto registry = torch::RegisterOperators().op("custom::ms_deform_attn", &ms_deform_attn);