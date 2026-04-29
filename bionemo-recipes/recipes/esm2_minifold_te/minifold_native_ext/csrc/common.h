#pragma once

#include <ATen/ATen.h>
#include <c10/util/Optional.h>

#include <string>
#include <tuple>

namespace minifold_native_ext {

inline c10::ScalarType parse_out_dtype(const std::string& dtype) {
  if (dtype == "float16") {
    return c10::ScalarType::Half;
  }
  if (dtype == "bfloat16") {
    return c10::ScalarType::BFloat16;
  }
  if (dtype == "float32") {
    return c10::ScalarType::Float;
  }
  throw std::invalid_argument("out_dtype must be one of: float16, bfloat16, float32");
}

std::tuple<at::Tensor, at::Tensor> linear_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& b_t,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias,
    const std::string& out_dtype,
    bool apply_relu = false,
    bool direct_fp8_output = false,
    bool fuse_bias_epilogue = false,
    const c10::optional<at::Tensor>& residual_payload = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale = c10::nullopt,
    const c10::optional<at::Tensor>& b_col_direct = c10::nullopt);

std::tuple<at::Tensor, at::Tensor, at::Tensor> linear_block32_fused_with_swizzled_scale_cuda(
    const at::Tensor& a,
    const at::Tensor& b_t,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias,
    const std::string& out_dtype,
    bool apply_relu = false,
    bool direct_fp8_output = false,
    bool fuse_bias_epilogue = false,
    const c10::optional<at::Tensor>& residual_payload = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale = c10::nullopt,
    const c10::optional<at::Tensor>& b_col_direct = c10::nullopt);

at::Tensor linear_block32_raw_debug_cuda(
    const at::Tensor& a,
    const at::Tensor& b_t,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias,
    const std::string& out_dtype,
    bool apply_relu = false,
    bool direct_fp8_output = false,
    bool fuse_bias_epilogue = false,
    const c10::optional<at::Tensor>& residual_payload = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale = c10::nullopt,
    const c10::optional<at::Tensor>& b_col_direct = c10::nullopt);

std::tuple<at::Tensor, at::Tensor> linear_block32_fc1_direct_cuda(
    const at::Tensor& a,
    const at::Tensor& b_cutlass_col,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& b_scale_swizzled,
    const at::Tensor& bias);

std::tuple<at::Tensor, at::Tensor> transition_norm_fc1_block32_fused_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& norm_weight,
    const at::Tensor& norm_bias,
    double norm_eps,
    const at::Tensor& b_cutlass_col,
    const at::Tensor& b_scale_swizzled,
    const c10::optional<at::Tensor>& bias = c10::nullopt);

at::Tensor transition_norm_fc1_bf16_fused_cuda(
    const at::Tensor& input,
    const at::Tensor& norm_weight,
    const at::Tensor& norm_bias,
    double norm_eps,
    const at::Tensor& fc1_weight,
    const c10::optional<at::Tensor>& fc1_bias = c10::nullopt);

at::Tensor transition_fc2_residual_bf16_fused_cuda(
    const at::Tensor& input,
    const at::Tensor& fc2_weight,
    const c10::optional<at::Tensor>& fc2_bias,
    const at::Tensor& residual);

std::tuple<at::Tensor, at::Tensor> gate_sigmoid_mul_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const std::string& out_dtype,
    const c10::optional<at::Tensor>& residual_payload = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale = c10::nullopt);

std::tuple<at::Tensor, at::Tensor, at::Tensor> gate_sigmoid_mul_block32_fused_with_swizzled_scale_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const std::string& out_dtype,
    const c10::optional<at::Tensor>& residual_payload = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale = c10::nullopt);

at::Tensor gate_sigmoid_mul_block32_raw_debug_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const std::string& out_dtype,
    const c10::optional<at::Tensor>& residual_payload = c10::nullopt,
    const c10::optional<at::Tensor>& residual_scale = c10::nullopt);

std::tuple<at::Tensor, at::Tensor> tri_mul_pair_from_block32_carrier_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& mask,
    const std::string& out_dtype);

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
pack_block32_to_mxfp8_fused_debug_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const c10::optional<at::Tensor>& mask = c10::nullopt);

std::tuple<at::Tensor, at::Tensor> tri_mul_pair_from_packed_debug_cuda(
    const at::Tensor& a1,
    const at::Tensor& b1,
    const at::Tensor& a2_t,
    const at::Tensor& b2_rhs,
    const at::Tensor& a1_scale_swizzled,
    const at::Tensor& b1_scale_swizzled,
    const at::Tensor& a2_t_scale_swizzled,
    const at::Tensor& b2_rhs_scale_swizzled,
    const std::string& out_dtype = "float16");

std::tuple<at::Tensor, at::Tensor> tri_pair_to_block32_carrier_debug_cuda(
    const at::Tensor& x1,
    const at::Tensor& x2,
    int64_t batch);

std::tuple<at::Tensor, at::Tensor> tri_gate_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const std::string& tri_out_dtype = "float16");

std::tuple<at::Tensor, at::Tensor> tri_input_norm_gate_block32_fused_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& input_norm_weight,
    const at::Tensor& input_norm_bias,
    double input_norm_eps,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const std::string& tri_out_dtype = "float16");

std::tuple<at::Tensor, at::Tensor> tri_gate_layernorm_block32_fused_cuda(
    const at::Tensor& a,
    const at::Tensor& a_scale_swizzled,
    const at::Tensor& lhs_b_t,
    const at::Tensor& lhs_scale_swizzled,
    const c10::optional<at::Tensor>& lhs_bias,
    const at::Tensor& rhs_b_t,
    const at::Tensor& rhs_scale_swizzled,
    const c10::optional<at::Tensor>& rhs_bias,
    const at::Tensor& mask,
    const at::Tensor& output_norm_weight,
    const at::Tensor& output_norm_bias,
    double output_norm_eps,
    const std::string& tri_out_dtype = "float16");

std::tuple<at::Tensor, at::Tensor> relu_block32_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale);

std::tuple<at::Tensor, at::Tensor> add_block32_cuda(
    const at::Tensor& lhs_payload,
    const at::Tensor& lhs_scale,
    const at::Tensor& rhs_payload,
    const at::Tensor& rhs_scale);

std::tuple<at::Tensor, at::Tensor> layernorm_block32_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps);

std::tuple<at::Tensor, at::Tensor, at::Tensor> layernorm_block32_with_swizzled_scale_cuda(
    const at::Tensor& payload,
    const at::Tensor& scale,
    const at::Tensor& weight,
    const at::Tensor& bias,
    double eps);

}  // namespace minifold_native_ext
