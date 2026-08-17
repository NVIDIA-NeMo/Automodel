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

"""Standalone fused rotary-position-embedding CUDA kernels.

The embedded device code is adapted from Transformer Engine commit
`366798ef8a0a00d8f2c1650d11e7e623d7c33e26`. It is compiled lazily as a
PyTorch extension and has no Transformer Engine runtime dependency.
"""

from __future__ import annotations

import functools
from typing import Literal, Protocol, cast

import torch
from torch.utils.cpp_extension import load_inline

_TensorFormat = Literal["sbhd", "bshd", "thd"]
_FORMAT_IDS: dict[_TensorFormat, int] = {"sbhd": 0, "bshd": 1, "thd": 2}

_CPP_SOURCE = r"""
#include <torch/extension.h>
torch::Tensor fused_rope_forward(torch::Tensor input, torch::Tensor freqs,
                                  torch::Tensor cu_seqlens, int64_t format,
                                  bool interleaved, int64_t cp_size, int64_t cp_rank,
                                  bool tokenwise_freqs);
torch::Tensor fused_rope_backward(torch::Tensor grad_output, torch::Tensor freqs,
                                   torch::Tensor cu_seqlens, int64_t format,
                                   bool interleaved, int64_t cp_size, int64_t cp_rank,
                                   bool tokenwise_freqs);
"""

_CUDA_SOURCE = r"""
// Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Adapted from Transformer Engine fused_rope.cu at commit
// 366798ef8a0a00d8f2c1650d11e7e623d7c33e26. Modified for standalone
// PyTorch bindings, input validation, and tokenwise packed frequencies.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <limits>

enum FusedRoPEFormat : int { SBHD = 0, BSHD = 1, THD = 2 };

template <typename scalar_t>
__device__ void fused_rope_block_forward(
    const scalar_t* src, const float* freqs, scalar_t* dst, const bool interleaved,
    const int s_id, const int offset_block, const int offset_block_dst,
    const int h, const int d, const int d2, const int stride_h,
    const int stride_d, const int o_stride_h, const int o_stride_d) {
  extern __shared__ float shared_mem_cos_sin[];
  float* shared_mem_cos = shared_mem_cos_sin;
  float* shared_mem_sin = shared_mem_cos_sin + d2;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
    sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
  }
  __syncthreads();
#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      float v_cos = shared_mem_cos[d_id];
      float v_sin = shared_mem_sin[d_id];
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_src_rotate;
      if (!interleaved) {
        v_src_rotate = (d_id + d2 / 2 < d2)
            ? -static_cast<float>(src[offset_src + (d2 / 2) * stride_d])
            : static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
      } else {
        v_src_rotate = (d_id % 2 == 0)
            ? -static_cast<float>(src[offset_src + stride_d])
            : static_cast<float>(src[offset_src - stride_d]);
      }
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
      }
    }
  }
}

template <typename scalar_t>
__device__ void fused_rope_block_backward(
    const scalar_t* src, const float* freqs, scalar_t* dst, const bool interleaved,
    const int s_id, const int offset_block, const int offset_block_dst,
    const int h, const int d, const int d2, const int stride_h,
    const int stride_d, const int o_stride_h, const int o_stride_d) {
  extern __shared__ float shared_mem_cos_sin[];
  float* shared_mem_cos = shared_mem_cos_sin;
  float* shared_mem_sin = shared_mem_cos_sin + d2;
  int tid = threadIdx.x * blockDim.y + threadIdx.y;
  for (int i = tid; i < d2; i += blockDim.x * blockDim.y) {
    sincosf(freqs[s_id * d2 + i], &shared_mem_sin[i], &shared_mem_cos[i]);
  }
  __syncthreads();
#pragma unroll
  for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
#pragma unroll
    for (int d_id = threadIdx.x; d_id < d2; d_id += blockDim.x) {
      int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
      int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
      float v_src = src[offset_src];
      float v_cos = shared_mem_cos[d_id];
      float v_src_rotate, v_sin;
      if (!interleaved) {
        if (d_id + d2 / 2 < d2) {
          v_src_rotate = static_cast<float>(src[offset_src + (d2 / 2) * stride_d]);
          v_sin = shared_mem_sin[d_id + d2 / 2];
        } else {
          v_src_rotate = static_cast<float>(src[offset_src + (d2 / 2 - d2) * stride_d]);
          v_sin = -shared_mem_sin[d_id + d2 / 2 - d2];
        }
      } else {
        if (d_id % 2 == 0) {
          v_src_rotate = static_cast<float>(src[offset_src + stride_d]);
          v_sin = shared_mem_sin[d_id + 1];
        } else {
          v_src_rotate = static_cast<float>(src[offset_src - stride_d]);
          v_sin = -shared_mem_sin[d_id - 1];
        }
      }
      dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin;
    }
  }
  if (d > d2) {
#pragma unroll
    for (int d_id = d2 + threadIdx.x; d_id < d; d_id += blockDim.x) {
#pragma unroll
      for (int h_id = threadIdx.y; h_id < h; h_id += blockDim.y) {
        int offset_src = offset_block + h_id * stride_h + d_id * stride_d;
        int offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d;
        dst[offset_dst] = src[offset_src];
      }
    }
  }
}

template <typename scalar_t>
__global__ void fused_rope_forward_kernel(
    const scalar_t* src, const int* cu_seqlens, const float* freqs,
    scalar_t* dst, const bool interleaved,
    const int cp_size, const int cp_rank, const bool tokenwise_freqs,
    const int s, const int h, const int d,
    const int d2, const int stride_s_or_t, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s_or_t, const int o_stride_b,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst, cur_seqlens;
  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }
  int s_id_for_freqs = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size
                        - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }
  if (tokenwise_freqs && cu_seqlens != nullptr) {
    s_id_for_freqs += cu_seqlens[b_id];
  }
  fused_rope_block_forward(src, freqs, dst, interleaved, s_id_for_freqs,
                           offset_block, offset_block_dst, h, d, d2, stride_h,
                           stride_d, o_stride_h, o_stride_d);
}

template <typename scalar_t>
__global__ void fused_rope_backward_kernel(
    const scalar_t* src, const int* cu_seqlens, const float* freqs,
    scalar_t* dst, const bool interleaved,
    const int cp_size, const int cp_rank, const bool tokenwise_freqs,
    const int s, const int h, const int d,
    const int d2, const int stride_s_or_t, const int stride_b, const int stride_h,
    const int stride_d, const int o_stride_s_or_t, const int o_stride_b,
    const int o_stride_h, const int o_stride_d) {
  int s_id = blockIdx.x, b_id = blockIdx.y;
  int offset_block, offset_block_dst, cur_seqlens;
  if (cu_seqlens != nullptr) {
    int start = cu_seqlens[b_id] / cp_size;
    int end = cu_seqlens[b_id + 1] / cp_size;
    int t_id = s_id + start;
    if (t_id >= end) return;
    offset_block = t_id * stride_s_or_t;
    offset_block_dst = t_id * o_stride_s_or_t;
    cur_seqlens = end - start;
  } else {
    offset_block = s_id * stride_s_or_t + b_id * stride_b;
    offset_block_dst = s_id * o_stride_s_or_t + b_id * o_stride_b;
    cur_seqlens = s;
  }
  int s_id_for_freqs = s_id;
  if (cp_size > 1) {
    assert(cur_seqlens % 2 == 0);
    if (s_id < cur_seqlens / 2) {
      s_id_for_freqs += cp_rank * cur_seqlens / 2;
    } else {
      s_id_for_freqs += cur_seqlens * cp_size
                        - (cp_rank + 1) * cur_seqlens / 2 - cur_seqlens / 2;
    }
  }
  if (tokenwise_freqs && cu_seqlens != nullptr) {
    s_id_for_freqs += cu_seqlens[b_id];
  }
  fused_rope_block_backward(src, freqs, dst, interleaved, s_id_for_freqs,
                            offset_block, offset_block_dst, h, d, d2, stride_h,
                            stride_d, o_stride_h, o_stride_d);
}

void check_args(const torch::Tensor& input, const torch::Tensor& freqs,
                const torch::Tensor& cu_seqlens, int format, int cp_size, int cp_rank,
                bool tokenwise_freqs) {
  TORCH_CHECK(input.is_cuda() && freqs.is_cuda() && cu_seqlens.is_cuda(), "CUDA tensors required");
  TORCH_CHECK(input.device() == freqs.device() && input.device() == cu_seqlens.device(),
              "input, freqs, and cu_seqlens must be on the same CUDA device");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(freqs.dim() == 4 && freqs.is_contiguous() && freqs.scalar_type() == at::kFloat,
              "freqs must be contiguous float32 [S,1,1,D2]");
  TORCH_CHECK(freqs.size(1) == 1 && freqs.size(2) == 1 && freqs.size(3) > 0 &&
              freqs.size(3) % 2 == 0,
              "freqs must have shape [positions, 1, 1, even rotary_dim]");
  TORCH_CHECK(input.numel() <= std::numeric_limits<int>::max() &&
              freqs.numel() <= std::numeric_limits<int>::max(),
              "input and freqs must each fit in 32-bit kernel indexing");
  TORCH_CHECK(format >= SBHD && format <= THD, "invalid format");
  TORCH_CHECK(cp_size >= 1 && cp_rank >= 0 && cp_rank < cp_size, "invalid CP configuration");
  TORCH_CHECK(input.scalar_type() == at::kFloat || input.scalar_type() == at::kHalf ||
              input.scalar_type() == at::kBFloat16, "supported dtypes: float32, float16, bfloat16");
  TORCH_CHECK(!tokenwise_freqs || format == THD,
              "tokenwise frequencies are only supported with THD input");
  if (format == THD) {
    TORCH_CHECK(input.dim() == 3, "THD input must be [T,H,D]");
    TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.scalar_type() == at::kInt &&
                cu_seqlens.is_contiguous(), "THD cu_seqlens must be contiguous int32");
    TORCH_CHECK(cu_seqlens.numel() >= 2, "THD cu_seqlens must contain at least [0, T]");

    // TE's fused kernel trusts these offsets and allocates output with empty_like.
    // Validate on the host before launch so incompatible real-token offsets cannot
    // leave padded physical-token slots unwritten and silently return garbage.
    auto cu_cpu = cu_seqlens.to(torch::kCPU);
    const int* offsets = cu_cpu.data_ptr<int>();
    const int64_t count = cu_cpu.numel();
    TORCH_CHECK(offsets[0] == 0, "THD cu_seqlens must start at 0");
    TORCH_CHECK(static_cast<int64_t>(offsets[count - 1]) == input.size(0) * cp_size,
                "incompatible THD metadata: cu_seqlens[-1] must equal "
                "input.size(0) * cp_size; pass cu_seqlens_padded for padded storage");
    for (int64_t i = 0; i + 1 < count; ++i) {
      const int64_t length = static_cast<int64_t>(offsets[i + 1]) - offsets[i];
      TORCH_CHECK(length >= 0, "THD cu_seqlens must be nondecreasing");
      if (!tokenwise_freqs) {
        TORCH_CHECK(length <= freqs.size(0),
                    "THD sequence length exceeds available rotary frequencies");
      }
      if (cp_size > 1) {
        TORCH_CHECK(length % (2 * cp_size) == 0,
                    "incompatible THD metadata: each padded sequence length must be "
                    "divisible by 2 * cp_size");
      }
    }
    if (tokenwise_freqs) {
      TORCH_CHECK(freqs.size(0) >= offsets[count - 1],
                  "tokenwise frequency table must cover every padded global THD token");
    }
  } else {
    TORCH_CHECK(input.dim() == 4, "SBHD/BSHD input must be rank 4");
  }
}

template <bool BACKWARD>
torch::Tensor launch(torch::Tensor input, torch::Tensor freqs, torch::Tensor cu_seqlens,
                     int format, bool interleaved, int cp_size, int cp_rank,
                     bool tokenwise_freqs) {
  check_args(input, freqs, cu_seqlens, format, cp_size, cp_rank, tokenwise_freqs);
  const c10::cuda::CUDAGuard guard(input.device());
  auto output = torch::empty_like(input);
  int s, b, h, d;
  if (format == THD) {
    s = freqs.size(0); b = cu_seqlens.size(0) - 1; h = input.size(1); d = input.size(2);
  } else if (format == SBHD) {
    s = input.size(0); b = input.size(1); h = input.size(2); d = input.size(3);
    TORCH_CHECK(s * cp_size <= freqs.size(0), "freqs too short for CP");
  } else {
    b = input.size(0); s = input.size(1); h = input.size(2); d = input.size(3);
    TORCH_CHECK(s * cp_size <= freqs.size(0), "freqs too short for CP");
  }
  TORCH_CHECK(format == THD || cp_size == 1 || s % 2 == 0,
              "local sequence length must be even when cp_size > 1");
  int d2 = freqs.size(3);
  TORCH_CHECK(d >= d2, "rotary dimension exceeds head dimension");
  int stride_s_or_t = format == SBHD ? input.stride(0) :
                      format == BSHD ? input.stride(1) : input.stride(0);
  int stride_b = format == SBHD ? input.stride(1) : format == BSHD ? input.stride(0) : 0;
  int out_stride_s_or_t = format == SBHD ? b * h * d : h * d;
  int out_stride_b = format == SBHD ? h * d : format == BSHD ? s * h * d : 0;
  int warps = h < 16 ? 4 : 8;
  dim3 blocks(s, b), threads(32, warps);
  int shared_mem = 2 * d2 * sizeof(float);
  const int* cu_ptr = format == THD ? cu_seqlens.data_ptr<int>() : nullptr;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), BACKWARD ? "fused_rope_backward" : "fused_rope_forward", [&] {
    if constexpr (BACKWARD) {
      fused_rope_backward_kernel<scalar_t><<<blocks, threads, shared_mem, stream>>>(
          input.data_ptr<scalar_t>(), cu_ptr, freqs.data_ptr<float>(),
          output.data_ptr<scalar_t>(), interleaved, cp_size, cp_rank, tokenwise_freqs,
          s, h, d, d2,
          stride_s_or_t, stride_b, input.stride(input.dim() - 2),
          input.stride(input.dim() - 1), out_stride_s_or_t, out_stride_b, d, 1);
    } else {
      fused_rope_forward_kernel<scalar_t><<<blocks, threads, shared_mem, stream>>>(
          input.data_ptr<scalar_t>(), cu_ptr, freqs.data_ptr<float>(),
          output.data_ptr<scalar_t>(), interleaved, cp_size, cp_rank, tokenwise_freqs,
          s, h, d, d2,
          stride_s_or_t, stride_b, input.stride(input.dim() - 2),
          input.stride(input.dim() - 1), out_stride_s_or_t, out_stride_b, d, 1);
    }
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor fused_rope_forward(torch::Tensor input, torch::Tensor freqs,
                                  torch::Tensor cu_seqlens, int64_t format,
                                  bool interleaved, int64_t cp_size, int64_t cp_rank,
                                  bool tokenwise_freqs) {
  return launch<false>(input, freqs, cu_seqlens, format, interleaved, cp_size, cp_rank,
                       tokenwise_freqs);
}
torch::Tensor fused_rope_backward(torch::Tensor grad_output, torch::Tensor freqs,
                                   torch::Tensor cu_seqlens, int64_t format,
                                   bool interleaved, int64_t cp_size, int64_t cp_rank,
                                   bool tokenwise_freqs) {
  return launch<true>(grad_output, freqs, cu_seqlens, format, interleaved, cp_size, cp_rank,
                      tokenwise_freqs);
}
"""


class _FusedRoPEExtension(Protocol):
    """Typed interface exposed by the lazily compiled extension."""

    def fused_rope_forward(
        self,
        input_tensor: torch.Tensor,
        freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tensor_format: int,
        interleaved: bool,
        cp_size: int,
        cp_rank: int,
        tokenwise_freqs: bool,
    ) -> torch.Tensor:
        """Launch the fused forward kernel.

        Args:
            input_tensor: Contiguous CUDA tensor of shape [sequence, batch, heads, head_dim],
                [batch, sequence, heads, head_dim], or [tokens, heads, head_dim].
            freqs: Contiguous float32 CUDA tensor of shape [positions, 1, 1, rotary_dim].
            cu_seqlens: Contiguous int32 CUDA tensor of shape [batch + 1] for THD input,
                or an empty tensor for padded input.
            tensor_format: Integer identifier for the input layout.
            interleaved: Whether rotary pairs occupy adjacent head-dimension elements.
            cp_size: Number of context-parallel shards.
            cp_rank: Rank of the local context-parallel shard.
            tokenwise_freqs: Whether THD frequencies are indexed by global physical token.

        Returns:
            Tensor with the same shape, dtype, and device as `input_tensor`.
        """
        ...

    def fused_rope_backward(
        self,
        grad_output: torch.Tensor,
        freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tensor_format: int,
        interleaved: bool,
        cp_size: int,
        cp_rank: int,
        tokenwise_freqs: bool,
    ) -> torch.Tensor:
        """Launch the fused input-gradient kernel.

        Args:
            grad_output: Contiguous CUDA tensor with the same layout, shape, and dtype
                as the corresponding forward output.
            freqs: Contiguous float32 CUDA tensor of shape [positions, 1, 1, rotary_dim].
            cu_seqlens: Contiguous int32 CUDA tensor of shape [batch + 1] for THD input,
                or an empty tensor for padded input.
            tensor_format: Integer identifier for the input layout.
            interleaved: Whether rotary pairs occupy adjacent head-dimension elements.
            cp_size: Number of context-parallel shards.
            cp_rank: Rank of the local context-parallel shard.
            tokenwise_freqs: Whether THD frequencies are indexed by global physical token.

        Returns:
            Input-gradient tensor with the same shape, dtype, and device as `grad_output`.
        """
        ...


@functools.cache
def _load_fused_rope_extension() -> _FusedRoPEExtension:
    """Build and cache the standalone extension for the visible CUDA architectures."""
    if not torch.cuda.is_available():
        raise RuntimeError("fused RoPE requires a CUDA-capable PyTorch installation")
    extension = load_inline(
        name="nemo_automodel_fused_rope_ext",
        cpp_sources=_CPP_SOURCE,
        cuda_sources=_CUDA_SOURCE,
        functions=["fused_rope_forward", "fused_rope_backward"],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=False,
    )
    return cast(_FusedRoPEExtension, extension)


class _FusedRoPEFunction(torch.autograd.Function):
    """Autograd bridge for the standalone fused RoPE kernels."""

    @staticmethod
    def forward(
        ctx,
        input_tensor: torch.Tensor,
        freqs: torch.Tensor,
        cu_seqlens: torch.Tensor,
        tensor_format: int,
        interleaved: bool,
        cp_size: int,
        cp_rank: int,
        tokenwise_freqs: bool,
    ) -> torch.Tensor:
        """Apply fused RoPE and retain immutable launch metadata for backward.

        Args:
            ctx: Autograd context for the current invocation.
            input_tensor: Contiguous CUDA tensor of shape [sequence, batch, heads, head_dim],
                [batch, sequence, heads, head_dim], or [tokens, heads, head_dim].
            freqs: Contiguous float32 CUDA tensor of shape [positions, 1, 1, rotary_dim].
            cu_seqlens: Contiguous int32 CUDA tensor of shape [batch + 1] for THD input,
                or an empty tensor for padded input.
            tensor_format: Integer identifier for the input layout.
            interleaved: Whether rotary pairs occupy adjacent head-dimension elements.
            cp_size: Number of context-parallel shards.
            cp_rank: Rank of the local context-parallel shard.
            tokenwise_freqs: Whether THD frequencies are indexed by global physical token.

        Returns:
            Tensor with the same shape, dtype, and device as `input_tensor`.
        """
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format
        ctx.interleaved = interleaved
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.tokenwise_freqs = tokenwise_freqs
        return _load_fused_rope_extension().fused_rope_forward(
            input_tensor,
            freqs,
            cu_seqlens,
            tensor_format,
            interleaved,
            cp_size,
            cp_rank,
            tokenwise_freqs,
        )

    @staticmethod
    def backward(
        ctx,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None, None]:
        """Apply the transposed rotary transform to the output gradient.

        Args:
            ctx: Autograd context containing the forward frequency tensor and launch metadata.
            grad_outputs: One-element tuple containing a CUDA tensor with the same semantic
                layout, shape, and dtype as the forward output. Arbitrary strides are accepted.

        Returns:
            Eight-element gradient tuple. The first element is a contiguous tensor with the
            same shape, dtype, and device as `grad_output`; metadata inputs have no gradients.
        """
        (grad_output,) = grad_outputs
        freqs, cu_seqlens = ctx.saved_tensors
        grad_input = _load_fused_rope_extension().fused_rope_backward(
            grad_output.contiguous(),
            freqs,
            cu_seqlens,
            ctx.tensor_format,
            ctx.interleaved,
            ctx.cp_size,
            ctx.cp_rank,
            ctx.tokenwise_freqs,
        )
        return grad_input, None, None, None, None, None, None, None


def apply_fused_rope(
    input_tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: _TensorFormat,
    interleaved: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    cp_size: int = 1,
    cp_rank: int = 0,
    tokenwise_freqs: bool = False,
) -> torch.Tensor:
    """Apply the standalone fused rotary-position-embedding CUDA kernel.

    For context parallelism, each local sequence must use the mirrored two-chunk
    layout: rank `r` owns global chunks `r` and `2 * cp_size - r - 1`.
    Frequencies contain raw angles, not precomputed cosine and sine values.

    Args:
        input_tensor: Contiguous CUDA tensor. SBHD uses shape [sequence, batch, heads,
            head_dim], BSHD uses [batch, sequence, heads, head_dim], and packed THD
            uses [tokens, heads, head_dim]. The final dimension may contain an
            unrotated suffix.
        freqs: Contiguous float32 CUDA tensor of shape [positions, 1, 1, rotary_dim].
            For half-split rotation, each angle appears in both halves; for
            interleaved rotation, each angle appears in adjacent pairs.
        tensor_format: Semantic layout of `input_tensor`: `"sbhd"`, `"bshd"`,
            or `"thd"`.
        interleaved: Whether rotary pairs occupy adjacent head-dimension elements.
        cu_seqlens: For THD, contiguous int32 CUDA offsets of shape [batch + 1]
            describing global padded sequence spans. The final offset must equal
            `tokens * cp_size`. Must be omitted for SBHD and BSHD.
        cp_size: Number of context-parallel shards.
        cp_rank: Rank of the local context-parallel shard.
        tokenwise_freqs: For THD, index `freqs` by global physical-token offset
            instead of restarting positions at zero for every packed sequence.

    Returns:
        New contiguous tensor with the same shape, dtype, and CUDA device as
        `input_tensor`. Only the rotary prefix is transformed.

    Raises:
        ValueError: If layout metadata is internally inconsistent or `freqs`
            requests gradients.
        RuntimeError: If CUDA is unavailable or a tensor violates the kernel's
            device, dtype, shape, contiguity, or indexing requirements.
    """
    if tensor_format not in _FORMAT_IDS:
        raise ValueError(f"unsupported tensor_format: {tensor_format}")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("cp_size must be positive and cp_rank must be in [0, cp_size)")
    if tokenwise_freqs and tensor_format != "thd":
        raise ValueError("tokenwise_freqs is only supported with THD input")
    if freqs.requires_grad:
        raise ValueError("fused RoPE does not compute frequency gradients")
    if not input_tensor.is_cuda or not freqs.is_cuda:
        raise RuntimeError("input_tensor and freqs must be CUDA tensors")
    if input_tensor.device != freqs.device:
        raise RuntimeError("input_tensor and freqs must be on the same CUDA device")

    if tensor_format == "thd":
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required with THD input")
        offsets = cu_seqlens
    else:
        if cu_seqlens is not None:
            raise ValueError("cu_seqlens must be omitted with SBHD and BSHD input")
        offsets = torch.empty(0, dtype=torch.int32, device=input_tensor.device)

    return _FusedRoPEFunction.apply(
        input_tensor,
        freqs,
        offsets,
        _FORMAT_IDS[tensor_format],
        interleaved,
        cp_size,
        cp_rank,
        tokenwise_freqs,
    )
