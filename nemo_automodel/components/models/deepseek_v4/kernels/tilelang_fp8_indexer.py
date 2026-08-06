# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""vLLM-compatible FP8 quantizer for DSV4 indexer Q and K activations."""

# ruff: noqa

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


@tilelang.jit(
    out_idx=[-2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def quantize_fp8_indexer(threads: int = 128):
    tokens = T.dynamic("tokens")

    @T.prim_func
    def main(
        Input: T.Tensor([tokens, 128], T.bfloat16),
        Values: T.Tensor([tokens, 128], T.float8_e4m3fn),
        Scales: T.Tensor([tokens, 1], T.float32),
    ):
        with T.Kernel(tokens, threads=threads) as token:
            values = T.alloc_fragment([128], T.bfloat16)
            amax = T.alloc_fragment([1], T.bfloat16)

            for offset in T.Parallel(128):
                values[offset] = Input[token, offset]
            T.reduce_absmax(values, amax, dim=0)

            # vLLM's post-RoPE indexer Q and indexer K-cache both use one
            # power-of-two scale per 128-element row:
            #   scale = 2**ceil(log2(max(abs(x), 1e-4) / 448)).
            descale = T.max(T.cast(amax[0], T.float32), 1.0e-4) / 448.0
            bits = T.reinterpret(descale, T.uint32)
            exponent = ((bits - 1) >> 23) + 1 - 127
            scale = T.reinterpret((exponent + 127) << 23, T.float32)
            scale_inv = T.reinterpret((127 - exponent) << 23, T.float32)

            for offset in T.Parallel(128):
                Values[token, offset] = T.cast(values[offset], T.float32) * scale_inv
            Scales[token, 0] = scale

    return main


def quantize_fp8_indexer_interface(x):
    """Quantize a contiguous ``[rows, 128]`` BF16 tensor."""
    assert x.is_cuda and x.dtype == torch.bfloat16
    assert x.ndim == 2 and x.shape[1] == 128 and x.is_contiguous()
    if x.shape[0] == 0:
        return (
            torch.empty((0, 128), dtype=torch.float8_e4m3fn, device=x.device),
            torch.empty((0, 1), dtype=torch.float32, device=x.device),
        )
    return quantize_fp8_indexer()(x)
