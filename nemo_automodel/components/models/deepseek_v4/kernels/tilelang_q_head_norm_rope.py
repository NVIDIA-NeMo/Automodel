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

"""Exact DSV4 per-head Q RMSNorm + interleaved RoPE TileLang kernel.

The forward reduction deliberately mirrors vLLM's
``fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel``:

* one 32-thread warp owns one 512-element query head;
* lane ``i`` owns the contiguous range ``[16*i, 16*i+16)``;
* each lane accumulates its 16 squares sequentially in FP32;
* the warp sum uses XOR shuffles in the order 16, 8, 4, 2, 1;
* normalization and the final 64-dimensional GPT-J RoPE remain in FP32 until
  the final BF16 store.

The custom backward is the analytic gradient of the same RMSNorm and rotation.
It intentionally uses regular PyTorch FP32 reductions: vLLM only defines the
inference-side forward boundary, while the backward must remain differentiable
for AutoModel training.
"""

# ruff: noqa

import torch

from nemo_automodel.components.models.deepseek_v4.kernels._tilelang import T, tilelang


_HEAD_DIM = 512
_ROPE_DIM = 64
_NOPE_DIM = _HEAD_DIM - _ROPE_DIM
_ELEMENTS_PER_LANE = _HEAD_DIM // 32


@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    },
)
def q_head_norm_rope_fwd(heads: int, eps: float, threads: int = 32):
    tokens = T.dynamic("tokens")

    @T.prim_func
    def main(
        Q: T.Tensor([tokens, heads, _HEAD_DIM], T.bfloat16),
        CosSin: T.Tensor([tokens, _ROPE_DIM], T.float32),
        Output: T.Tensor([tokens, heads, _HEAD_DIM], T.bfloat16),
    ):
        with T.Kernel(tokens, heads, threads=threads) as (token, head):
            lane = T.get_thread_binding()
            sum_squares = T.alloc_var(T.float32)
            sum_squares = 0.0

            # vLLM assigns 16 adjacent values to each lane.  Keep this serial
            # accumulation order: a generic 512-way reduction changes a few
            # final BF16 values and the following FP8 boundary amplifies them.
            for offset in T.serial(_ELEMENTS_PER_LANE):
                value = T.cast(Q[token, head, lane * _ELEMENTS_PER_LANE + offset], T.float32)
                sum_squares += value * value

            sum_squares += T.shfl_xor(sum_squares, 16)
            sum_squares += T.shfl_xor(sum_squares, 8)
            sum_squares += T.shfl_xor(sum_squares, 4)
            sum_squares += T.shfl_xor(sum_squares, 2)
            sum_squares += T.shfl_xor(sum_squares, 1)
            rms_rcp = T.rsqrt(sum_squares / 512.0 + eps)

            if lane < _NOPE_DIM // _ELEMENTS_PER_LANE:
                for offset in T.serial(_ELEMENTS_PER_LANE):
                    dim = lane * _ELEMENTS_PER_LANE + offset
                    Output[token, head, dim] = T.cast(T.cast(Q[token, head, dim], T.float32) * rms_rcp, T.bfloat16)
            else:
                rope_base = lane * _ELEMENTS_PER_LANE - _NOPE_DIM
                for pair in T.serial(_ELEMENTS_PER_LANE // 2):
                    local_dim = rope_base + pair * 2
                    dim = _NOPE_DIM + local_dim
                    cos_value = CosSin[token, local_dim // 2]
                    sin_value = CosSin[token, _ROPE_DIM // 2 + local_dim // 2]
                    even = T.cast(Q[token, head, dim], T.float32) * rms_rcp
                    odd = T.cast(Q[token, head, dim + 1], T.float32) * rms_rcp
                    Output[token, head, dim] = T.cast(even * cos_value - odd * sin_value, T.bfloat16)
                    Output[token, head, dim + 1] = T.cast(even * sin_value + odd * cos_value, T.bfloat16)

    return main


class _QHeadNormRope(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        cos_sin: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        output = q_head_norm_rope_fwd(q.shape[1], eps)(q, cos_sin)
        ctx.save_for_backward(q, cos_sin)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None]:
        q, cos_sin = ctx.saved_tensors
        q_float = q.float()
        grad_norm = grad_output.float().clone()

        cos = cos_sin[:, None, : _ROPE_DIM // 2]
        sin = cos_sin[:, None, _ROPE_DIM // 2 :]
        grad_pairs = grad_output[..., -_ROPE_DIM:].float().unflatten(-1, (-1, 2))
        grad_even = grad_pairs[..., 0]
        grad_odd = grad_pairs[..., 1]
        norm_pairs = grad_norm[..., -_ROPE_DIM:].unflatten(-1, (-1, 2))
        norm_pairs[..., 0] = grad_even * cos + grad_odd * sin
        norm_pairs[..., 1] = -grad_even * sin + grad_odd * cos

        rms_rcp = torch.rsqrt(q_float.square().mean(dim=-1, keepdim=True) + ctx.eps)
        projection = (grad_norm * q_float).mean(dim=-1, keepdim=True)
        grad_q = grad_norm * rms_rcp - q_float * projection * rms_rcp.pow(3)
        return grad_q.to(q.dtype), None, None


def q_head_norm_rope_interface(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_head_dim: int,
    eps: float,
) -> torch.Tensor:
    """Run exact vLLM-compatible Q normalization and RoPE on ``[B,H,S,512]``."""
    if not (q.is_cuda and q.dtype == torch.bfloat16):
        raise RuntimeError("DSV4 Q head norm/RoPE TileLang kernel requires a CUDA BF16 query")
    if q.ndim != 4 or q.shape[-1] != _HEAD_DIM or rope_head_dim != _ROPE_DIM:
        raise ValueError(
            "DSV4 Q head norm/RoPE TileLang kernel requires [B,H,S,512] "
            f"with rope_head_dim=64, got {tuple(q.shape)} and {rope_head_dim}"
        )

    batch, heads, seq_len, head_dim = q.shape
    if q.numel() == 0:
        return q
    if cos.shape[:2] != (batch, seq_len) or sin.shape[:2] != (batch, seq_len):
        raise ValueError(
            "DSV4 Q head norm/RoPE expected cos/sin [B,S,*] matching q; "
            f"got q={tuple(q.shape)}, cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )

    token_major_q = q.transpose(1, 2).contiguous().view(batch * seq_len, heads, head_dim)
    cos_sin = (
        torch.cat(
            (
                cos[..., : _ROPE_DIM // 2].float(),
                sin[..., : _ROPE_DIM // 2].float(),
            ),
            dim=-1,
        )
        .contiguous()
        .view(batch * seq_len, _ROPE_DIM)
    )
    output = _QHeadNormRope.apply(token_major_q, cos_sin, eps)
    return output.view(batch, seq_len, heads, head_dim).transpose(1, 2)
