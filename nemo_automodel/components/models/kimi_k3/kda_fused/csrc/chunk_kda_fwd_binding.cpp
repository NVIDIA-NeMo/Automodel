// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Fused chunked Kimi Delta Attention forward for the Kimi-K3 call. One CTA per (head, document)
// walks the 64-token chunks with the [K, V] state on-chip (mma.sync + ldmatrix, sm_80+). Specialised to head dims
// K = V = 128, a packed layout (batch 1 with int32 cu_seqlens), q/k l2-normalised in kernel, a bounded log gate, no
// initial/final state. Measured on GB200 against FLA 0.4.2 chunk_kda at the Kimi-K3 layer-microbatch shape
// (8192 tokens = 4 documents, 96 heads): 0.78-0.80 ms vs 2.07-2.19 ms, max |diff| 1-2 bf16 ulp.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include "chunk_kda_fwd.h"

namespace {
constexpr int64_t kHeadDim = 128;

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& g,
                  const torch::Tensor& beta, const torch::Tensor& cu_seqlens, const torch::Tensor& o) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda() && cu_seqlens.is_cuda() &&
                  o.is_cuda(),
              "chunk_kda_fwd: all tensors must live on the GPU");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1, "chunk_kda_fwd: q must be [1, T, H, K] (packed batch of 1), got ",
              q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() && g.sizes() == q.sizes(),
              "chunk_kda_fwd: k, v, g must match q's shape [1, T, H, 128]; got k ", k.sizes(), " v ", v.sizes(), " g ",
              g.sizes());
  TORCH_CHECK(q.size(3) == kHeadDim, "chunk_kda_fwd: head dims must be K = V = 128, got ", q.size(3));
  TORCH_CHECK(beta.dim() == 3 && beta.size(0) == 1 && beta.size(1) == q.size(1) && beta.size(2) == q.size(2),
              "chunk_kda_fwd: beta must be [1, T, H], got ", beta.sizes());
  TORCH_CHECK(o.sizes() == v.sizes(), "chunk_kda_fwd: o must match v's shape, got ", o.sizes());
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                  v.scalar_type() == at::kBFloat16 && o.scalar_type() == at::kBFloat16,
              "chunk_kda_fwd: q, k, v, o must be bfloat16");
  TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
              "chunk_kda_fwd: g and beta must be float32");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
              "chunk_kda_fwd: cu_seqlens must be a 1-D int32 tensor of at least two offsets");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() &&
                  beta.is_contiguous() && cu_seqlens.is_contiguous() && o.is_contiguous(),
              "chunk_kda_fwd: all tensors must be contiguous");
}
}  // namespace

void run(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
         const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& cu_seqlens,
         torch::Tensor& o) {
  check_inputs(q, k, v, g, beta, cu_seqlens, o);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  int H = (int)q.size(2);
  int ndoc = (int)cu_seqlens.size(0) - 1;
  cudaError_t err = kda_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                               cu_seqlens.data_ptr(), o.data_ptr(), H, ndoc,
                               at::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "chunk_kda_fwd: cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ",
              cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "chunk_kda forward (CUDA, K = V = 128, packed batch of 1)");
}
