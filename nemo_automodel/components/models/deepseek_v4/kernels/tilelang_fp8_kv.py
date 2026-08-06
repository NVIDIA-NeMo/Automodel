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

"""TileLang quantizer for vLLM's ``fp8_ds_mla`` numerical format.

The 512-dimensional DSV4 KV vector is stored as seven 64-element E4M3
NoPE groups with UE8M0 (power-of-two) descales, followed by 64 BF16 RoPE
elements.  An eighth scale byte is retained as padding to match vLLM's
per-token scale stride.
"""

# ruff: noqa

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


@tilelang.jit(
    out_idx=[-3, -2, -1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def quantize_fp8_ds_mla(threads: int = 128):
    tokens = T.dynamic("tokens")
    input_shape = [tokens, 512]
    nope_shape = [tokens, 448]
    rope_shape = [tokens, 64]
    scale_shape = [tokens, 8]

    @T.prim_func
    def main(
        KV: T.Tensor(input_shape, T.bfloat16),
        Nope: T.Tensor(nope_shape, T.float8_e4m3fn),
        Rope: T.Tensor(rope_shape, T.bfloat16),
        Scales: T.Tensor(scale_shape, T.uint8),
    ):
        with T.Kernel(tokens, threads=threads) as token:
            nope = T.alloc_fragment([7, 64], T.bfloat16)
            amax = T.alloc_fragment([7], T.bfloat16)
            scale_inv = T.alloc_fragment([7], T.float32)

            for group, offset in T.Parallel(7, 64):
                nope[group, offset] = KV[token, group * 64 + offset]
            T.reduce_absmax(nope, amax, dim=1)

            for group in T.Parallel(7):
                # vLLM clamps amax before computing ceil(log2(amax / 448)).
                descale = T.max(T.cast(amax[group], T.float32), 1.0e-4) / 448.0
                bits = T.reinterpret(descale, T.uint32)
                exponent = ((bits - 1) >> 23) + 1 - 127
                Scales[token, group] = T.cast(exponent + 127, T.uint8)
                scale_inv[group] = T.reinterpret((127 - exponent) << 23, T.float32)

            for group, offset in T.Parallel(7, 64):
                Nope[token, group * 64 + offset] = T.cast(nope[group, offset], T.float32) * scale_inv[group]
            for offset in T.Parallel(64):
                Rope[token, offset] = KV[token, 448 + offset]
            Scales[token, 7] = T.cast(0, T.uint8)

    return main


def quantize_fp8_ds_mla_interface(kv):
    """Quantize a contiguous ``[tokens, 512]`` BF16 tensor."""
    assert kv.is_cuda and kv.dtype == torch.bfloat16
    assert kv.ndim == 2 and kv.shape[1] == 512 and kv.is_contiguous()
    if kv.shape[0] == 0:
        return (
            torch.empty((0, 448), dtype=torch.float8_e4m3fn, device=kv.device),
            torch.empty((0, 64), dtype=torch.bfloat16, device=kv.device),
            torch.empty((0, 8), dtype=torch.uint8, device=kv.device),
        )
    return quantize_fp8_ds_mla()(kv)
