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

#pragma once
#include <cuda_runtime.h>
// Returns the status of the one-time cudaFuncSetAttribute (dynamic shared memory); the launch itself is checked by
// the caller (C10_CUDA_KERNEL_LAUNCH_CHECK in the binding).
cudaError_t kda_launch(const void* q, const void* k, const void* v, const void* g, const void* beta,
                       const void* cu, void* o, int H, int ndoc, cudaStream_t stream);
