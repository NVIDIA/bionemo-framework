#include "common.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

#include <cmath>

namespace minifold_native_ext {

namespace {

void check_cuda_bf16_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == c10::ScalarType::BFloat16, name, " must have dtype torch.bfloat16");
}

void check_optional_cuda_bf16_tensor(const c10::optional<at::Tensor>& t, const char* name) {
  if (!t.has_value()) {
    return;
  }
  check_cuda_bf16_tensor(*t, name);
}

void set_cuda_device(const at::Tensor& t) {
  TORCH_CHECK(cudaSetDevice(static_cast<int>(t.get_device())) == cudaSuccess, "failed to set CUDA device");
}

inline cudaStream_t current_cuda_stream() {
  return static_cast<cudaStream_t>(0);
}

void check_kernel_launch(const char* name) {
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, name, " launch failed: ", cudaGetErrorString(err));
}

__device__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void transition_norm_to_bf16_input_kernel(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* output,
    int rows,
    float eps) {
  constexpr int kCols = 128;
  constexpr int kGroups = 4;
  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 4 + warp_id;
  if (warp_id >= 4 || row >= rows) {
    return;
  }

  float x_vals[kGroups];
  float sum = 0.0f;
  float sumsq = 0.0f;
#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    const float x = __bfloat162float(input[row * kCols + col]);
    x_vals[group] = x;
    sum += x;
    sumsq += x * x;
  }

  const float total = warp_reduce_sum(sum);
  const float total_sq = warp_reduce_sum(sumsq);
  const float mean = __shfl_sync(0xffffffff, total / static_cast<float>(kCols), 0);
  const float inv_std = __shfl_sync(
      0xffffffff,
      rsqrtf(fmaxf(total_sq / static_cast<float>(kCols) - mean * mean, 0.0f) + eps),
      0);

#pragma unroll
  for (int group = 0; group < kGroups; ++group) {
    const int col = lane + group * 32;
    const float y =
        (x_vals[group] - mean) * inv_std * __bfloat162float(weight[col]) + __bfloat162float(bias[col]);
    output[row * kCols + col] = __float2bfloat16(y);
  }
}

__global__ void bias_relu_bf16_512_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* bias,
    int rows) {
  constexpr int kCols = 512;
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= rows || tid >= 256) {
    return;
  }

  const int base = row * kCols;
  const int col0 = tid;
  float value0 = __bfloat162float(output[base + col0]);
  if (bias != nullptr) {
    value0 += __bfloat162float(bias[col0]);
  }
  if (value0 < 0.0f) {
    value0 = 0.0f;
  }
  output[base + col0] = __float2bfloat16(value0);

  const int col1 = tid + 256;
  float value1 = __bfloat162float(output[base + col1]);
  if (bias != nullptr) {
    value1 += __bfloat162float(bias[col1]);
  }
  if (value1 < 0.0f) {
    value1 = 0.0f;
  }
  output[base + col1] = __float2bfloat16(value1);
}

__global__ void bias_residual_bf16_128_kernel(
    __nv_bfloat16* output,
    const __nv_bfloat16* bias,
    const __nv_bfloat16* residual,
    int rows) {
  constexpr int kCols = 128;
  const int row = blockIdx.x;
  const int col = threadIdx.x;
  if (row >= rows || col >= kCols) {
    return;
  }

  const int idx = row * kCols + col;
  float value = __bfloat162float(output[idx]) + __bfloat162float(residual[idx]);
  if (bias != nullptr) {
    value += __bfloat162float(bias[col]);
  }
  output[idx] = __float2bfloat16(value);
}

}  // namespace

at::Tensor transition_norm_fc1_bf16_fused_cuda(
    const at::Tensor& input,
    const at::Tensor& norm_weight,
    const at::Tensor& norm_bias,
    double norm_eps,
    const at::Tensor& fc1_weight,
    const c10::optional<at::Tensor>& fc1_bias) {
  check_cuda_bf16_tensor(input, "input");
  check_cuda_bf16_tensor(norm_weight, "norm_weight");
  check_cuda_bf16_tensor(norm_bias, "norm_bias");
  check_cuda_bf16_tensor(fc1_weight, "fc1_weight");
  check_optional_cuda_bf16_tensor(fc1_bias, "fc1_bias");
  TORCH_CHECK(input.dim() == 3, "transition_norm_fc1_bf16_fused expects input with shape [1, rows, 128]");
  TORCH_CHECK(input.size(0) == 1 && input.size(2) == 128, "transition_norm_fc1_bf16_fused expects input width 128");
  TORCH_CHECK(norm_weight.dim() == 1 && norm_weight.size(0) == 128, "norm_weight must have shape [128]");
  TORCH_CHECK(norm_bias.dim() == 1 && norm_bias.size(0) == 128, "norm_bias must have shape [128]");
  TORCH_CHECK(fc1_weight.sizes() == at::IntArrayRef({512, 128}), "fc1_weight must have shape [512, 128]");
  TORCH_CHECK(!fc1_bias.has_value() || fc1_bias->sizes() == at::IntArrayRef({512}), "fc1_bias must have shape [512]");
  TORCH_CHECK(
      input.device() == norm_weight.device() && input.device() == norm_bias.device() && input.device() == fc1_weight.device() &&
          (!fc1_bias.has_value() || input.device() == fc1_bias->device()),
      "all tensors must be on the same CUDA device");

  set_cuda_device(input);
  const int rows = static_cast<int>(input.size(1));
  auto normed = at::empty_like(input);
  const int blocks = static_cast<int>((rows + 3) / 4);
  transition_norm_to_bf16_input_kernel<<<blocks, 128, 0, current_cuda_stream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(norm_weight.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(norm_bias.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(normed.data_ptr<at::BFloat16>()),
      rows,
      static_cast<float>(norm_eps));
  check_kernel_launch("transition_norm_to_bf16_input_kernel");

  auto normed_2d = normed.view({rows, 128});
  auto output_2d = at::matmul(normed_2d, fc1_weight.transpose(0, 1));
  auto output = output_2d.view({1, rows, 512}).contiguous();
  const __nv_bfloat16* bias_ptr = nullptr;
  if (fc1_bias.has_value()) {
    bias_ptr = reinterpret_cast<const __nv_bfloat16*>(fc1_bias->data_ptr<at::BFloat16>());
  }
  bias_relu_bf16_512_kernel<<<rows, 256, 0, current_cuda_stream()>>>(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      bias_ptr,
      rows);
  check_kernel_launch("bias_relu_bf16_512_kernel");
  return output;
}

at::Tensor transition_fc2_residual_bf16_fused_cuda(
    const at::Tensor& input,
    const at::Tensor& fc2_weight,
    const c10::optional<at::Tensor>& fc2_bias,
    const at::Tensor& residual) {
  check_cuda_bf16_tensor(input, "input");
  check_cuda_bf16_tensor(fc2_weight, "fc2_weight");
  check_optional_cuda_bf16_tensor(fc2_bias, "fc2_bias");
  check_cuda_bf16_tensor(residual, "residual");
  TORCH_CHECK(input.dim() == 3, "transition_fc2_residual_bf16_fused expects input with shape [1, rows, 512]");
  TORCH_CHECK(
      input.size(0) == 1 && input.size(2) == 512,
      "transition_fc2_residual_bf16_fused expects input width 512");
  TORCH_CHECK(
      residual.dim() == 3 && residual.size(0) == 1 && residual.size(1) == input.size(1) && residual.size(2) == 128,
      "residual must have shape [1, rows, 128]");
  TORCH_CHECK(fc2_weight.sizes() == at::IntArrayRef({128, 512}), "fc2_weight must have shape [128, 512]");
  TORCH_CHECK(!fc2_bias.has_value() || fc2_bias->sizes() == at::IntArrayRef({128}), "fc2_bias must have shape [128]");
  TORCH_CHECK(
      input.device() == residual.device() && input.device() == fc2_weight.device() &&
          (!fc2_bias.has_value() || input.device() == fc2_bias->device()),
      "all tensors must be on the same CUDA device");

  set_cuda_device(input);
  const int rows = static_cast<int>(input.size(1));
  auto input_2d = input.view({rows, 512});
  auto output_2d = at::matmul(input_2d, fc2_weight.transpose(0, 1));
  auto output = output_2d.view({1, rows, 128}).contiguous();
  const __nv_bfloat16* bias_ptr = nullptr;
  if (fc2_bias.has_value()) {
    bias_ptr = reinterpret_cast<const __nv_bfloat16*>(fc2_bias->data_ptr<at::BFloat16>());
  }
  bias_residual_bf16_128_kernel<<<rows, 128, 0, current_cuda_stream()>>>(
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      bias_ptr,
      reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<at::BFloat16>()),
      rows);
  check_kernel_launch("bias_residual_bf16_128_kernel");
  return output;
}

}  // namespace minifold_native_ext
