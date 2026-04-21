#include <torch/extension.h>

#include "common.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "linear_block32_fused",
      &minifold_native_ext::linear_block32_fused_cuda,
      "MiniFold native MXFP8 linear forward with fused block32 output quantization",
      pybind11::arg("a"),
      pybind11::arg("b_t"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("b_scale_swizzled"),
      pybind11::arg("bias") = pybind11::none(),
      pybind11::arg("out_dtype") = "bfloat16",
      pybind11::arg("apply_relu") = false,
      pybind11::arg("direct_fp8_output") = false,
      pybind11::arg("fuse_bias_epilogue") = false,
      pybind11::arg("residual_payload") = pybind11::none(),
      pybind11::arg("residual_scale") = pybind11::none());
  m.def(
      "linear_block32_raw_debug",
      &minifold_native_ext::linear_block32_raw_debug_cuda,
      "Debug-only native MXFP8 linear forward returning dense BF16 output before block32 carrier requantization",
      pybind11::arg("a"),
      pybind11::arg("b_t"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("b_scale_swizzled"),
      pybind11::arg("bias") = pybind11::none(),
      pybind11::arg("out_dtype") = "bfloat16",
      pybind11::arg("apply_relu") = false,
      pybind11::arg("direct_fp8_output") = false,
      pybind11::arg("fuse_bias_epilogue") = false,
      pybind11::arg("residual_payload") = pybind11::none(),
      pybind11::arg("residual_scale") = pybind11::none());
  m.def(
      "linear_block32_fc1_direct",
      &minifold_native_ext::linear_block32_fc1_direct_cuda,
      "MiniFold native canonical fc1 path with direct FP8 block32 output",
      pybind11::arg("a"),
      pybind11::arg("b_cutlass_col"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("b_scale_swizzled"),
      pybind11::arg("bias"));
  m.def(
      "transition_norm_fc1_block32_fused",
      &minifold_native_ext::transition_norm_fc1_block32_fused_cuda,
      "MiniFold native fused transition layernorm plus fc1 MXFP8 GEMM",
      pybind11::arg("payload"),
      pybind11::arg("scale"),
      pybind11::arg("norm_weight"),
      pybind11::arg("norm_bias"),
      pybind11::arg("norm_eps"),
      pybind11::arg("b_cutlass_col"),
      pybind11::arg("b_scale_swizzled"),
      pybind11::arg("bias") = pybind11::none());
  m.def(
      "transition_norm_fc1_bf16_fused",
      &minifold_native_ext::transition_norm_fc1_bf16_fused_cuda,
      "MiniFold native fused BF16 transition layernorm plus fc1",
      pybind11::arg("input"),
      pybind11::arg("norm_weight"),
      pybind11::arg("norm_bias"),
      pybind11::arg("norm_eps"),
      pybind11::arg("fc1_weight"),
      pybind11::arg("fc1_bias") = pybind11::none());
  m.def(
      "transition_fc2_residual_bf16_fused",
      &minifold_native_ext::transition_fc2_residual_bf16_fused_cuda,
      "MiniFold native fused BF16 transition fc2 plus residual add",
      pybind11::arg("input"),
      pybind11::arg("fc2_weight"),
      pybind11::arg("fc2_bias") = pybind11::none(),
      pybind11::arg("residual"));
  m.def(
      "gate_sigmoid_mul_block32_fused",
      &minifold_native_ext::gate_sigmoid_mul_block32_fused_cuda,
      "MiniFold native fused gate path: two MXFP8 GEMMs plus sigmoid-mul block32 output",
      pybind11::arg("a"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("lhs_b_t"),
      pybind11::arg("lhs_scale_swizzled"),
      pybind11::arg("lhs_bias") = pybind11::none(),
      pybind11::arg("rhs_b_t"),
      pybind11::arg("rhs_scale_swizzled"),
      pybind11::arg("rhs_bias") = pybind11::none(),
      pybind11::arg("out_dtype") = "bfloat16",
      pybind11::arg("residual_payload") = pybind11::none(),
      pybind11::arg("residual_scale") = pybind11::none());
  m.def(
      "gate_sigmoid_mul_block32_raw_debug",
      &minifold_native_ext::gate_sigmoid_mul_block32_raw_debug_cuda,
      "Debug-only native fused gate path returning dense BF16 output before block32 carrier requantization",
      pybind11::arg("a"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("lhs_b_t"),
      pybind11::arg("lhs_scale_swizzled"),
      pybind11::arg("lhs_bias") = pybind11::none(),
      pybind11::arg("rhs_b_t"),
      pybind11::arg("rhs_scale_swizzled"),
      pybind11::arg("rhs_bias") = pybind11::none(),
      pybind11::arg("out_dtype") = "bfloat16",
      pybind11::arg("residual_payload") = pybind11::none(),
      pybind11::arg("residual_scale") = pybind11::none());
  m.def(
      "tri_mul_pair_from_block32_carrier",
      &minifold_native_ext::tri_mul_pair_from_block32_carrier_cuda,
      "MiniFold native tri path from resident block32 carrier to resident block32 carrier",
      pybind11::arg("payload"),
      pybind11::arg("scale"),
      pybind11::arg("mask") = pybind11::none(),
      pybind11::arg("out_dtype") = "float16");
  m.def(
      "pack_block32_to_mxfp8_fused_debug",
      &minifold_native_ext::pack_block32_to_mxfp8_fused_debug_cuda,
      "Debug-only native pack from resident block32 carrier to MXFP8 tri operands",
      pybind11::arg("payload"),
      pybind11::arg("scale"),
      pybind11::arg("mask") = pybind11::none());
  m.def(
      "tri_mul_pair_from_packed_debug",
      &minifold_native_ext::tri_mul_pair_from_packed_debug_cuda,
      "Debug-only native raw tri GEMM pair on prepacked MXFP8 operands",
      pybind11::arg("a1"),
      pybind11::arg("b1"),
      pybind11::arg("a2_t"),
      pybind11::arg("b2_rhs"),
      pybind11::arg("a1_scale_swizzled"),
      pybind11::arg("b1_scale_swizzled"),
      pybind11::arg("a2_t_scale_swizzled"),
      pybind11::arg("b2_rhs_scale_swizzled"),
      pybind11::arg("out_dtype") = "float16");
  m.def(
      "tri_pair_to_block32_carrier_debug",
      &minifold_native_ext::tri_pair_to_block32_carrier_debug_cuda,
      "Debug-only native repack from raw tri pair outputs back to resident block32 carrier",
      pybind11::arg("x1"),
      pybind11::arg("x2"),
      pybind11::arg("batch"));
  m.def(
      "tri_gate_block32_fused",
      &minifold_native_ext::tri_gate_block32_fused_cuda,
      "MiniFold native fused pre-tri gate pack + raw tri GEMMs to resident block32 carrier",
      pybind11::arg("a"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("lhs_b_t"),
      pybind11::arg("lhs_scale_swizzled"),
      pybind11::arg("lhs_bias") = pybind11::none(),
      pybind11::arg("rhs_b_t"),
      pybind11::arg("rhs_scale_swizzled"),
      pybind11::arg("rhs_bias") = pybind11::none(),
      pybind11::arg("mask"),
      pybind11::arg("tri_out_dtype") = "float16");
  m.def(
      "tri_input_norm_gate_block32_fused",
      &minifold_native_ext::tri_input_norm_gate_block32_fused_cuda,
      "MiniFold native fused tri input layernorm + gate pack + raw tri GEMMs to resident block32 carrier",
      pybind11::arg("payload"),
      pybind11::arg("scale"),
      pybind11::arg("input_norm_weight"),
      pybind11::arg("input_norm_bias"),
      pybind11::arg("input_norm_eps"),
      pybind11::arg("lhs_b_t"),
      pybind11::arg("lhs_scale_swizzled"),
      pybind11::arg("lhs_bias") = pybind11::none(),
      pybind11::arg("rhs_b_t"),
      pybind11::arg("rhs_scale_swizzled"),
      pybind11::arg("rhs_bias") = pybind11::none(),
      pybind11::arg("mask"),
      pybind11::arg("tri_out_dtype") = "float16");
  m.def(
      "tri_gate_layernorm_block32_fused",
      &minifold_native_ext::tri_gate_layernorm_block32_fused_cuda,
      "MiniFold native fused pre-tri gate + raw tri GEMMs + post-tri layernorm to resident block32 carrier",
      pybind11::arg("a"),
      pybind11::arg("a_scale_swizzled"),
      pybind11::arg("lhs_b_t"),
      pybind11::arg("lhs_scale_swizzled"),
      pybind11::arg("lhs_bias") = pybind11::none(),
      pybind11::arg("rhs_b_t"),
      pybind11::arg("rhs_scale_swizzled"),
      pybind11::arg("rhs_bias") = pybind11::none(),
      pybind11::arg("mask"),
      pybind11::arg("output_norm_weight"),
      pybind11::arg("output_norm_bias"),
      pybind11::arg("output_norm_eps"),
      pybind11::arg("tri_out_dtype") = "float16");
  m.def(
      "relu_block32",
      &minifold_native_ext::relu_block32_cuda,
      "MiniFold native block32 relu",
      pybind11::arg("payload"),
      pybind11::arg("scale"));
  m.def(
      "add_block32",
      &minifold_native_ext::add_block32_cuda,
      "MiniFold native block32 add",
      pybind11::arg("lhs_payload"),
      pybind11::arg("lhs_scale"),
      pybind11::arg("rhs_payload"),
      pybind11::arg("rhs_scale"));
  m.def(
      "layernorm_block32",
      &minifold_native_ext::layernorm_block32_cuda,
      "MiniFold native block32 layernorm",
      pybind11::arg("payload"),
      pybind11::arg("scale"),
      pybind11::arg("weight"),
      pybind11::arg("bias"),
      pybind11::arg("eps"));
}
