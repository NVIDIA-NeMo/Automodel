# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Source text of the Adam step of NVFP4 expert weights (CUDA) and its torch binding.

``nvfp4_resident_adam`` hands these strings to ``torch.utils.cpp_extension.load_inline`` as
``cpp_sources = [HEADER, BINDING]`` and ``cuda_sources = [HEADER, KERNEL]``, so the wheel ships Python files only.
The header is inlined into both translation units (there is no file to ``#include``), which is why it carries no
``#pragma once``.

Both kernels return the bits of the Transformer Engine kernel they stand in for. The Adam arithmetic is written as
explicit PTX operations for that reason: Transformer Engine builds its Adam kernel with ``--use_fast_math``, and
the result depends on which operations the compiler then emits.
"""

_LICENSE = r"""// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// Licensed under the Apache License, Version 2.0; see http://www.apache.org/licenses/LICENSE-2.0.
"""

HEADER = (
    _LICENSE
    + r"""
#include <cstdint>
#include <cuda_runtime.h>

// Tensors of one launch. Kernel arguments are limited to 4 KiB, which bounds this count.
constexpr int kMaxTensors = 32;
constexpr int kAdamThreads = 256;
constexpr int kAdamElementsPerCta = kAdamThreads * 4 * 16;
constexpr int kTile = 128;  // matrix elements per thread-block side of the requantization; one thread per 16x16 block

// One Adam launch: per tensor the BF16 gradient, the FP32 master weight and both FP32 moments.
struct AdamBatch {
  const uint16_t* grad[kMaxTensors];
  float* master[kMaxTensors];
  float* exp_avg[kMaxTensors];
  float* exp_avg_sq[kMaxTensors];
  int64_t numel[kMaxTensors];
  int32_t cta_end[kMaxTensors];  // running total of thread blocks
  float* amax;                   // [count] largest magnitude of each updated master weight
  int32_t count;
};

void launch_adam(const AdamBatch& batch, int total_ctas, float lr, float beta1, float beta2, float beta1_correction,
                 float beta2_correction, float epsilon, int mode, float decay, cudaStream_t stream);

// One requantization launch: per tensor the FP32 master weight and its row-wise and column-wise NVFP4 operands.
struct RequantizeBatch {
  const float* master[kMaxTensors];
  uint8_t* row_data[kMaxTensors];   // [rows, cols / 2]
  uint8_t* row_scale[kMaxTensors];  // [>= rows, row_scale_stride]
  uint8_t* col_data[kMaxTensors];   // [cols, rows / 2]
  uint8_t* col_scale[kMaxTensors];  // [>= cols, col_scale_stride]
  float* amax_row[kMaxTensors];
  float* amax_col[kMaxTensors];
  int32_t rows[kMaxTensors];
  int32_t cols[kMaxTensors];
  int32_t row_scale_stride[kMaxTensors];
  int32_t col_scale_stride[kMaxTensors];
  int32_t cta_end[kMaxTensors];  // running total of thread blocks
  const float* amax;             // [count] largest magnitude of each master weight
  int32_t count;
};

void launch_requantize(const RequantizeBatch& batch, int total_ctas, cudaStream_t stream);
"""
)

KERNEL = (
    _LICENSE
    + r"""
// Adam on FP32 master weights with FP32 moments and BF16 gradients. Each thread updates four consecutive elements
// per iteration, and every thread block reports the largest magnitude of the master elements it wrote.

#include <cfloat>

namespace {

// Transformer Engine builds its Adam kernel (common/multi_tensor/adam.cu) with --use_fast_math, under which the
// compiler and the assembler pick which products to fuse and use approximate division and square root. The
// operations below are the ones they emit for AdamFunctor, written out with explicit rounding so that no compiler
// option or code shape changes the result.
__device__ __forceinline__ float mul_ftz(const float a, const float b) {
  float d;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float add_ftz(const float a, const float b) {
  float d;
  asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float sub_ftz(const float a, const float b) {
  float d;
  asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float fma_ftz(const float a, const float b, const float c) {
  float d;
  asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(a), "f"(b), "f"(c));
  return d;
}
__device__ __forceinline__ float div_approx_ftz(const float a, const float b) {
  float d;
  asm("div.approx.ftz.f32 %0, %1, %2;" : "=f"(d) : "f"(a), "f"(b));
  return d;
}
__device__ __forceinline__ float sqrt_approx_ftz(const float a) {
  float d;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(d) : "f"(a));
  return d;
}

__device__ __forceinline__ float bf16_bits_to_float(uint32_t bits) { return __uint_as_float(bits << 16); }

struct AdamScalars {
  float lr, beta1, beta2, one_minus_beta1, one_minus_beta2, beta1_correction, beta2_correction, epsilon, decay;
  int mode;  // 0: L2 regularization, 1: decoupled weight decay
};

__device__ __forceinline__ void adam_element(float g, float& p, float& m, float& v, const AdamScalars& s) {
  if (s.mode == 0) g = fma_ftz(s.decay, p, g);
  m = fma_ftz(s.beta1, m, mul_ftz(s.one_minus_beta1, g));
  v = fma_ftz(s.beta2, v, mul_ftz(mul_ftz(s.one_minus_beta2, g), g));
  const float m_unbiased = div_approx_ftz(m, s.beta1_correction);
  const float v_unbiased = div_approx_ftz(v, s.beta2_correction);
  float update = div_approx_ftz(m_unbiased, add_ftz(sqrt_approx_ftz(v_unbiased), s.epsilon));
  if (s.mode != 0) update = fma_ftz(s.decay, p, update);
  p = fma_ftz(-s.lr, update, p);
}

__global__ void __launch_bounds__(kAdamThreads) adam_kernel(const AdamBatch batch, const float lr, const float beta1,
                                                            const float beta2, const float beta1_correction,
                                                            const float beta2_correction, const float epsilon,
                                                            const int mode, const float decay) {
  int tensor = 0;
  while (static_cast<int>(blockIdx.x) >= batch.cta_end[tensor]) ++tensor;
  const int cta_begin = tensor == 0 ? 0 : batch.cta_end[tensor - 1];
  const int64_t offset = static_cast<int64_t>(blockIdx.x - cta_begin) * kAdamElementsPerCta;
  const int64_t remaining = batch.numel[tensor] - offset;
  const uint16_t* const g = batch.grad[tensor] + offset;
  float* const p = batch.master[tensor] + offset;
  float* const m = batch.exp_avg[tensor] + offset;
  float* const v = batch.exp_avg_sq[tensor] + offset;
  const AdamScalars s{lr, beta1, beta2, sub_ftz(1.0f, beta1), sub_ftz(1.0f, beta2), beta1_correction,
                      beta2_correction, epsilon, decay, mode};

  float thread_amax = 0.0f;
  for (int64_t i = static_cast<int64_t>(threadIdx.x) * 4; i < remaining && i < kAdamElementsPerCta;
       i += kAdamThreads * 4) {
    const uint64_t g4 = *reinterpret_cast<const uint64_t*>(g + i);
    float4 r_p = *reinterpret_cast<const float4*>(p + i);
    float4 r_m = *reinterpret_cast<const float4*>(m + i);
    float4 r_v = *reinterpret_cast<const float4*>(v + i);
    adam_element(bf16_bits_to_float(static_cast<uint32_t>(g4) & 0xffffu), r_p.x, r_m.x, r_v.x, s);
    adam_element(bf16_bits_to_float(static_cast<uint32_t>(g4 >> 16) & 0xffffu), r_p.y, r_m.y, r_v.y, s);
    adam_element(bf16_bits_to_float(static_cast<uint32_t>(g4 >> 32) & 0xffffu), r_p.z, r_m.z, r_v.z, s);
    adam_element(bf16_bits_to_float(static_cast<uint32_t>(g4 >> 48) & 0xffffu), r_p.w, r_m.w, r_v.w, s);
    *reinterpret_cast<float4*>(p + i) = r_p;
    *reinterpret_cast<float4*>(m + i) = r_m;
    *reinterpret_cast<float4*>(v + i) = r_v;
    thread_amax = fmaxf(thread_amax, fmaxf(fmaxf(fabsf(r_p.x), fabsf(r_p.y)), fmaxf(fabsf(r_p.z), fabsf(r_p.w))));
  }
  // Largest magnitude of this thread block, then of the tensor. Non-negative floats order like their bit patterns.
  for (int delta = 16; delta >= 1; delta /= 2) {
    thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xffffffffu, thread_amax, delta));
  }
  __shared__ float warp_amax[kAdamThreads / 32];
  if ((threadIdx.x & 31) == 0) warp_amax[threadIdx.x / 32] = thread_amax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float cta_amax = 0.0f;
    for (int w = 0; w < kAdamThreads / 32; ++w) cta_amax = fmaxf(cta_amax, warp_amax[w]);
    atomicMax(reinterpret_cast<unsigned int*>(batch.amax + tensor), __float_as_uint(cta_amax));
  }
}


// 2D NVFP4 quantization of FP32 master weights into the row-wise and the column-wise operand of a Transformer
// Engine NVFP4 weight. The arithmetic is that of Transformer Engine's quantize_transpose_nvfp4_2D_kernel
// (common/cast/nvfp4): one E4M3 scale per 16x16 block from the block and tensor maxima, then each element times the
// reciprocal of the decoded scale, rounded to E2M1. Both operands hold the same codes; the column-wise one is the
// transpose.

// FP32 to E4M3 byte: round to nearest even, saturate at 448. `x` is non-negative.
__device__ __forceinline__ uint8_t e4m3_encode(const float x) {
  if (!(x > 0.0f)) return 0;
  const uint32_t bits = __float_as_uint(x);
  int exponent = static_cast<int>(bits >> 23) - 127;
  if (exponent < -6) return static_cast<uint8_t>(rintf(x * 512.0f));  // subnormal range, quantum 2^-9
  uint32_t mantissa = (bits >> 20) & 7u;
  const uint32_t rest = bits & 0xfffffu;
  if (rest > 0x80000u || (rest == 0x80000u && (mantissa & 1u))) {
    if (++mantissa == 8u) {
      mantissa = 0u;
      ++exponent;
    }
  }
  if (exponent > 8 || (exponent == 8 && mantissa == 7u)) return 0x7e;
  return static_cast<uint8_t>(((exponent + 7) << 3) | mantissa);
}

__device__ __forceinline__ float e4m3_decode(const uint8_t byte) {
  const uint32_t exponent = (byte >> 3) & 15u;
  const uint32_t mantissa = byte & 7u;
  if (exponent == 0u) return static_cast<float>(mantissa) * (1.0f / 512.0f);
  return static_cast<float>(8u + mantissa) * static_cast<float>(1u << exponent) * (1.0f / 1024.0f);
}

// FP32 to the 4-bit E2M1 code (sign in bit 3): round to nearest even, saturate at 6.
__device__ __forceinline__ uint32_t e2m1_encode(const float y) {
  const float a = fabsf(y);
  const uint32_t magnitude = static_cast<uint32_t>(a > 0.25f) + static_cast<uint32_t>(a >= 0.75f) +
                             static_cast<uint32_t>(a > 1.25f) + static_cast<uint32_t>(a >= 1.75f) +
                             static_cast<uint32_t>(a > 2.5f) + static_cast<uint32_t>(a >= 3.5f) +
                             static_cast<uint32_t>(a > 5.0f);
  return magnitude | ((__float_as_uint(y) >> 28) & 8u);
}

// A thread block covers a 128x128 tile of one weight; each of its 64 threads quantizes one 16x16 block.
__global__ void __launch_bounds__(64) requantize_kernel(const RequantizeBatch batch) {
  int tensor = 0;
  while (static_cast<int>(blockIdx.x) >= batch.cta_end[tensor]) ++tensor;
  const int local_cta = blockIdx.x - (tensor == 0 ? 0 : batch.cta_end[tensor - 1]);
  const int rows = batch.rows[tensor];
  const int cols = batch.cols[tensor];
  const float amax = batch.amax[tensor];
  if (local_cta == 0 && threadIdx.x == 0) {
    *batch.amax_row[tensor] = amax;
    *batch.amax_col[tensor] = amax;
  }
  const int tiles_x = (cols + kTile - 1) / kTile;
  const int r0 = (local_cta / tiles_x) * kTile + static_cast<int>(threadIdx.x >> 3) * 16;
  const int c0 = (local_cta % tiles_x) * kTile + static_cast<int>(threadIdx.x & 7) * 16;
  if (r0 >= rows || c0 >= cols) return;

  float s_enc = fminf(448.0f * 6.0f / amax, FLT_MAX);
  if (amax == 0.0f || s_enc == 0.0f) s_enc = 1.0f;
  const float s_dec = 1.0 / s_enc;

  const float* const src = batch.master[tensor] + static_cast<int64_t>(r0) * cols + c0;
  float block_amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float4* const row = reinterpret_cast<const float4*>(src + static_cast<int64_t>(i) * cols);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float4 x = row[q];
      block_amax = fmaxf(block_amax, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
    }
  }
  constexpr float fp4_max_inv = 1.0f / 6.0f;
  const float s_dec_b = block_amax * (s_enc * fp4_max_inv);
  const uint8_t scale = e4m3_encode(fminf(s_dec_b, FLT_MAX));
  const float inverse = fminf(1.0f / (e4m3_decode(scale) * s_dec), FLT_MAX);

  const int64_t row_bytes = cols / 2;
  const int64_t col_bytes = rows / 2;
  const int64_t row_scale_stride = batch.row_scale_stride[tensor];
  const int64_t col_scale_stride = batch.col_scale_stride[tensor];
  uint8_t* const row_data = batch.row_data[tensor] + static_cast<int64_t>(r0) * row_bytes + c0 / 2;
  uint8_t* const row_scale = batch.row_scale[tensor] + static_cast<int64_t>(r0) * row_scale_stride + c0 / 16;
  uint64_t col_words[16];
#pragma unroll
  for (int j = 0; j < 16; ++j) col_words[j] = 0;
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    const float4* const row = reinterpret_cast<const float4*>(src + static_cast<int64_t>(i) * cols);
    uint64_t row_word = 0;
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float4 x = row[q];
      const uint32_t code0 = e2m1_encode(x.x * inverse);
      const uint32_t code1 = e2m1_encode(x.y * inverse);
      const uint32_t code2 = e2m1_encode(x.z * inverse);
      const uint32_t code3 = e2m1_encode(x.w * inverse);
      row_word |= static_cast<uint64_t>(code0 | (code1 << 4) | (code2 << 8) | (code3 << 12)) << (16 * q);
      col_words[4 * q + 0] |= static_cast<uint64_t>(code0) << (4 * i);
      col_words[4 * q + 1] |= static_cast<uint64_t>(code1) << (4 * i);
      col_words[4 * q + 2] |= static_cast<uint64_t>(code2) << (4 * i);
      col_words[4 * q + 3] |= static_cast<uint64_t>(code3) << (4 * i);
    }
    *reinterpret_cast<uint64_t*>(row_data + i * row_bytes) = row_word;
    row_scale[i * row_scale_stride] = scale;
  }
  uint8_t* const col_data = batch.col_data[tensor] + static_cast<int64_t>(c0) * col_bytes + r0 / 2;
  uint8_t* const col_scale = batch.col_scale[tensor] + static_cast<int64_t>(c0) * col_scale_stride + r0 / 16;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    *reinterpret_cast<uint64_t*>(col_data + j * col_bytes) = col_words[j];
    col_scale[j * col_scale_stride] = scale;
  }
}

}  // namespace

void launch_adam(const AdamBatch& batch, int total_ctas, float lr, float beta1, float beta2, float beta1_correction,
                 float beta2_correction, float epsilon, int mode, float decay, cudaStream_t stream) {
  adam_kernel<<<total_ctas, kAdamThreads, 0, stream>>>(batch, lr, beta1, beta2, beta1_correction, beta2_correction,
                                                      epsilon, mode, decay);
}

void launch_requantize(const RequantizeBatch& batch, int total_ctas, cudaStream_t stream) {
  requantize_kernel<<<total_ctas, 64, 0, stream>>>(batch);
}
"""
)

BINDING = (
    _LICENSE
    + r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

using TensorList = std::vector<torch::Tensor>;

void check_buffer(const torch::Tensor& t, at::ScalarType dtype, int64_t numel, size_t alignment, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "nvfp4 adam: ", name, " must be a contiguous CUDA tensor");
  TORCH_CHECK(t.scalar_type() == dtype, "nvfp4 adam: ", name, " has dtype ", t.scalar_type(), ", expected ", dtype);
  TORCH_CHECK(t.numel() >= numel, "nvfp4 adam: ", name, " holds ", t.numel(), " elements, expected at least ", numel);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % alignment == 0, "nvfp4 adam: ", name, " is not ",
              alignment, "-byte aligned");
}

}  // namespace

// One Adam step on the FP32 master weights [rows, cols] of NVFP4 weights, with FP32 moments and BF16 gradients,
// then both NVFP4 operands of every weight rewritten from its master. `mode` is 0 for L2 regularization and 1 for
// decoupled weight decay. Block scales are [rows, cols / 16] and [cols, rows / 16] bytes, each possibly padded;
// the largest magnitude of every master goes to both amax tensors of its weight.
void adam_requantize(const TensorList& grads, const TensorList& masters, const TensorList& exp_avgs,
                     const TensorList& exp_avg_sqs, const TensorList& row_data, const TensorList& row_scales,
                     const TensorList& col_data, const TensorList& col_scales, const TensorList& amax_rows,
                     const TensorList& amax_cols, const float lr, const float beta1, const float beta2,
                     const float epsilon, const int step, const int mode, const int bias_correction,
                     const float weight_decay) {
  const size_t count = masters.size();
  TORCH_CHECK(count > 0 && grads.size() == count && exp_avgs.size() == count && exp_avg_sqs.size() == count &&
                  row_data.size() == count && row_scales.size() == count && col_data.size() == count &&
                  col_scales.size() == count && amax_rows.size() == count && amax_cols.size() == count,
              "nvfp4 adam: the tensor lists must be non-empty and have one entry per weight");
  const c10::cuda::OptionalCUDAGuard guard(masters[0].device());
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // The bias corrections of Transformer Engine's multi_tensor_adam_cuda.
  float bias_correction1 = 1.0f, bias_correction2 = 1.0f;
  if (bias_correction == 1) {
    bias_correction1 = 1 - std::pow(beta1, step);
    bias_correction2 = 1 - std::pow(beta2, step);
  }
  torch::Tensor amax = torch::zeros({static_cast<int64_t>(count)}, masters[0].options().dtype(at::kFloat));

  for (size_t first = 0; first < count; first += kMaxTensors) {
    AdamBatch adam;
    RequantizeBatch requantize;
    adam.count = requantize.count = static_cast<int>(std::min<size_t>(kMaxTensors, count - first));
    adam.amax = amax.data_ptr<float>() + first;
    requantize.amax = adam.amax;
    int adam_ctas = 0, requantize_ctas = 0;
    for (int i = 0; i < adam.count; ++i) {
      const size_t t = first + i;
      const torch::Tensor& master = masters[t];
      TORCH_CHECK(master.dim() == 2 && master.size(0) % 16 == 0 && master.size(1) % 16 == 0,
                  "nvfp4 adam: master weights must be matrices with both sides divisible by 16, got ",
                  master.sizes());
      const int64_t rows = master.size(0), cols = master.size(1), numel = rows * cols;
      check_buffer(master, at::kFloat, numel, 16, "master weight");
      check_buffer(exp_avgs[t], at::kFloat, numel, 16, "first moment");
      check_buffer(exp_avg_sqs[t], at::kFloat, numel, 16, "second moment");
      check_buffer(grads[t], at::kBFloat16, numel, 8, "gradient");
      TORCH_CHECK(exp_avgs[t].numel() == numel && exp_avg_sqs[t].numel() == numel && grads[t].numel() == numel,
                  "nvfp4 adam: gradient and moments must have the master weight's element count ", numel);
      check_buffer(row_data[t], at::kByte, numel / 2, 8, "row-wise data");
      check_buffer(col_data[t], at::kByte, numel / 2, 8, "column-wise data");
      TORCH_CHECK(row_scales[t].dim() == 2 && row_scales[t].size(0) >= rows && row_scales[t].size(1) >= cols / 16,
                  "nvfp4 adam: row-wise scales have shape ", row_scales[t].sizes());
      TORCH_CHECK(col_scales[t].dim() == 2 && col_scales[t].size(0) >= cols && col_scales[t].size(1) >= rows / 16,
                  "nvfp4 adam: column-wise scales have shape ", col_scales[t].sizes());
      check_buffer(row_scales[t], at::kByte, rows * (cols / 16), 1, "row-wise scales");
      check_buffer(col_scales[t], at::kByte, cols * (rows / 16), 1, "column-wise scales");
      check_buffer(amax_rows[t], at::kFloat, 1, 4, "row-wise amax");
      check_buffer(amax_cols[t], at::kFloat, 1, 4, "column-wise amax");

      adam.grad[i] = static_cast<const uint16_t*>(grads[t].data_ptr());
      adam.master[i] = master.data_ptr<float>();
      adam.exp_avg[i] = exp_avgs[t].data_ptr<float>();
      adam.exp_avg_sq[i] = exp_avg_sqs[t].data_ptr<float>();
      adam.numel[i] = numel;
      adam_ctas += static_cast<int>((numel + kAdamElementsPerCta - 1) / kAdamElementsPerCta);
      adam.cta_end[i] = adam_ctas;

      requantize.master[i] = master.data_ptr<float>();
      requantize.row_data[i] = row_data[t].data_ptr<uint8_t>();
      requantize.row_scale[i] = row_scales[t].data_ptr<uint8_t>();
      requantize.col_data[i] = col_data[t].data_ptr<uint8_t>();
      requantize.col_scale[i] = col_scales[t].data_ptr<uint8_t>();
      requantize.amax_row[i] = amax_rows[t].data_ptr<float>();
      requantize.amax_col[i] = amax_cols[t].data_ptr<float>();
      requantize.rows[i] = static_cast<int32_t>(rows);
      requantize.cols[i] = static_cast<int32_t>(cols);
      requantize.row_scale_stride[i] = static_cast<int32_t>(row_scales[t].size(1));
      requantize.col_scale_stride[i] = static_cast<int32_t>(col_scales[t].size(1));
      requantize_ctas += static_cast<int>(((rows + kTile - 1) / kTile) * ((cols + kTile - 1) / kTile));
      requantize.cta_end[i] = requantize_ctas;
    }
    launch_adam(adam, adam_ctas, lr, beta1, beta2, bias_correction1, bias_correction2, epsilon, mode, weight_decay,
                stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    launch_requantize(requantize, requantize_ctas, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("adam_requantize", &adam_requantize, "Adam step on FP32 master weights and NVFP4 write-back");
}
"""
)
